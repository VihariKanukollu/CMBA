from typing import Tuple, List, Dict
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class HierarchicalReasoningModel4L_ACTV2InnerCarry:
    z_C: torch.Tensor
    z_M: torch.Tensor
    z_B: torch.Tensor
    z_A: torch.Tensor


@dataclass
class HierarchicalReasoningModel4L_ACTV2Carry:
    inner_carry: HierarchicalReasoningModel4L_ACTV2InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel4L_ACTV2Config(BaseModel):
    # Input/config
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    # Macro-cycle counts (A cycles) for ACT outer loop compatibility
    A_cycles: int

    # Timescale hierarchy parameters
    M_per_B: int = 4
    B_per_A: int = 2
    C_every_A: int = 2  # Update C every N A steps

    # Transformer depth per level
    C_layers: int
    M_layers: int
    B_layers: int
    A_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    # Vedic control/gating
    use_film: bool = True
    num_chitta_slots: int = 1  # number of leading C tokens used as slow memory slots
    ego_token: bool = True     # use first A token as ego head source
    num_plan_tokens: int = 0   # optional planning tokens prepended to B

    forward_dtype: str = "bfloat16"
    
    # Ablations/toggles
    freeze_c_writes: bool = False
    disable_c_read_in_b: bool = False


class HierarchicalReasoningModel4L_ACTV2Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel4L_ACTV2Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel4L_ACTV2ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel4L_ACTV2Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class HierarchicalReasoningModel4L_ACTV2_Inner(nn.Module):
    """V2 inner model with Vedic-rooted timescales and A-driven FiLM/gating.

    Key differences from v1:
      - Raw inputs inject only into M. B attends to (M + C_read). A updates from B.
      - A produces FiLM parameters (alpha/beta) for M and B, and a write gate for C.
      - C updates slowly via EMA-like writes to the leading num_chitta_slots tokens.
      - Nested schedule: M_per_B micro-steps, B_per_A steps per A, C writes every C_every_A A steps.
    """

    def __init__(self, config: HierarchicalReasoningModel4L_ACTV2Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self._last_g_C_write = None
        self._last_alpha_M = None
        self._last_alpha_B = None

        # I/O
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # Positional encodings
        if self.config.pos_encodings == "rope":
            # Set RoPE capacity to cover the largest sequence length we may pass to any block
            base_len = self.config.seq_len + self.puzzle_emb_len
            max_extra = self.config.num_chitta_slots + (1 if self.config.ego_token else 0) + self.config.num_plan_tokens + base_len  # for B context concat with M
            rope_max = base_len + max_extra
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=rope_max,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            # For v2, we need extra positions for chitta slots, ego token, and plan tokens
            max_extra = self.config.num_chitta_slots + (1 if self.config.ego_token else 0) + self.config.num_plan_tokens
            total_positions = self.config.seq_len + self.puzzle_emb_len + max_extra
            self.embed_pos = CastedEmbedding(total_positions, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
            # WARNING: With learned positions, extra tokens (ego, plan, chitta slots) need manual position injection
            # Consider using RoPE instead for v2 architecture
        else:
            raise NotImplementedError()

        # Reasoning layers per level
        self.C_level = HierarchicalReasoningModel4L_ACTV2ReasoningModule(layers=[HierarchicalReasoningModel4L_ACTV2Block(self.config) for _i in range(self.config.C_layers)])
        self.M_level = HierarchicalReasoningModel4L_ACTV2ReasoningModule(layers=[HierarchicalReasoningModel4L_ACTV2Block(self.config) for _i in range(self.config.M_layers)])
        self.B_level = HierarchicalReasoningModel4L_ACTV2ReasoningModule(layers=[HierarchicalReasoningModel4L_ACTV2Block(self.config) for _i in range(self.config.B_layers)])
        self.A_level = HierarchicalReasoningModel4L_ACTV2ReasoningModule(layers=[HierarchicalReasoningModel4L_ACTV2Block(self.config) for _i in range(self.config.A_layers)])
        
        # Initial states
        self.C_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.M_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.B_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.A_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # FiLM controller from A (produces alpha/beta for M and B, and C write gate)
        film_out_dim = 4 * self.config.hidden_size + 1  # alpha_M, beta_M, alpha_B, beta_B, g_C_write
        self.film_mlp = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(self.config.hidden_size, film_out_dim, bias=True)
        )
        # Initialize last layer to near-identity effect: alphas default to 1, betas/gate to 0
        with torch.no_grad():
            if isinstance(self.film_mlp[-1], nn.Linear):
                self.film_mlp[-1].weight.zero_()
                self.film_mlp[-1].bias.zero_()

        # Candidate writer for C from pooled (A, B, M)
        self.c_write_proj = nn.Linear(self.config.hidden_size * 3, self.config.hidden_size, bias=True)
        with torch.no_grad():
            self.c_write_proj.weight.zero_()
            self.c_write_proj.bias.zero_()

        # Q head special init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        # Get device from one of the model parameters
        device = next(self.parameters()).device
        base_len = self.config.seq_len + self.puzzle_emb_len
        len_C = base_len + max(1, self.config.num_chitta_slots)
        len_M = base_len
        len_B = base_len + max(0, self.config.num_plan_tokens)
        len_A = base_len + (1 if self.config.ego_token else 0)
        return HierarchicalReasoningModel4L_ACTV2InnerCarry(
            z_C=torch.empty(batch_size, len_C, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_M=torch.empty(batch_size, len_M, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_B=torch.empty(batch_size, len_B, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_A=torch.empty(batch_size, len_A, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel4L_ACTV2InnerCarry):
        return HierarchicalReasoningModel4L_ACTV2InnerCarry(
            z_C=torch.where(reset_flag.view(-1, 1, 1), self.C_init, carry.z_C),
            z_M=torch.where(reset_flag.view(-1, 1, 1), self.M_init, carry.z_M),
            z_B=torch.where(reset_flag.view(-1, 1, 1), self.B_init, carry.z_B),
            z_A=torch.where(reset_flag.view(-1, 1, 1), self.A_init, carry.z_A),
        )

    @staticmethod
    def _mean_pool(hidden_states: torch.Tensor) -> torch.Tensor:
        # [batch, seq, hidden] -> [batch, hidden]
        return hidden_states.mean(dim=-2)

    def _film_from_A(self, z_A: torch.Tensor):
        # Pool A then compute gating
        pooled_A = self._mean_pool(z_A).to(torch.float32)
        film_raw = self.film_mlp(pooled_A)
        hs = self.config.hidden_size
        alpha_M_raw, beta_M_raw, alpha_B_raw, beta_B_raw, gC_raw = torch.split(film_raw, [hs, hs, hs, hs, 1], dim=-1)
        # Alphas default to 1.0
        alpha_M = 1.0 + alpha_M_raw
        alpha_B = 1.0 + alpha_B_raw
        # Betas default to 0.0
        beta_M = beta_M_raw
        beta_B = beta_B_raw
        # Gate defaults to small (bias zero -> 0.5, shift to be small)
        g_C_write = torch.sigmoid(gC_raw - 4.0)  # ~0.018 initially
        return alpha_M, beta_M, alpha_B, beta_B, g_C_write

    @staticmethod
    def _apply_film(injection: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # Broadcast alpha/beta [batch, hidden] to [batch, seq, hidden]
        # Cast to injection dtype to avoid upcasting large tensors to float32
        d = injection.dtype
        alpha_b = alpha.to(d).unsqueeze(1)
        beta_b = beta.to(d).unsqueeze(1)
        return alpha_b * injection + beta_b

    def _c_read_summary(self, z_C: torch.Tensor) -> torch.Tensor:
        # Use the first num_chitta_slots tokens as memory slots; pool them
        slots = z_C[:, :max(1, self.config.num_chitta_slots)]
        return slots.mean(dim=-2)  # [batch, hidden]

    def _c_write_update(self, z_C: torch.Tensor, z_A: torch.Tensor, z_B: torch.Tensor, z_M: torch.Tensor, g_C_write: torch.Tensor) -> torch.Tensor:
        # Compute candidate write from pooled A, B, M
        pooled = torch.cat([self._mean_pool(z_A), self._mean_pool(z_B), self._mean_pool(z_M)], dim=-1).to(torch.float32)
        candidate = self.c_write_proj(pooled)  # [batch, hidden]
        # Broadcast to slots
        slots = z_C[:, :max(1, self.config.num_chitta_slots)]
        g = g_C_write.clamp(0, 1).unsqueeze(1)  # [batch, 1, 1]
        candidate_b = candidate.unsqueeze(1).to(slots.dtype)
        updated_slots = (1.0 - g) * slots + g * candidate_b
        z_C = torch.cat([updated_slots, z_C[:, max(1, self.config.num_chitta_slots):]], dim=-2)
        return z_C

    def forward(self, carry: HierarchicalReasoningModel4L_ACTV2InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel4L_ACTV2InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        def seq_info_for(length: int):
            if hasattr(self, "rotary_emb"):
                cos, sin = self.rotary_emb()
                return dict(cos_sin=(cos[:length], sin[:length]))
            return dict(cos_sin=None)

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations (no-grad settle)
        with torch.no_grad():
            z_C, z_M, z_B, z_A = carry.z_C, carry.z_M, carry.z_B, carry.z_A

            for _a_step in range(self.config.A_cycles):
                alpha_M, beta_M, alpha_B, beta_B, g_C_write = self._film_from_A(z_A)

                for _b_step in range(self.config.B_per_A):
                    # M micro-steps
                    for _m_step in range(self.config.M_per_B):
                        # Raw inputs only to M; FiLM modulates injection based on A
                        inj_M = input_embeddings
                        if self.config.use_film:
                            inj_M = self._apply_film(inj_M, alpha_M, beta_M)
                        z_M = self.M_level(z_M, inj_M, **seq_info_for(z_M.shape[-2]))

                    # B integrates M and C read via context concatenation [B | M | C_slots]
                    base_len = input_embeddings.shape[-2]
                    plan_tokens = max(0, self.config.num_plan_tokens)
                    c_slots = z_C[:, :max(1, self.config.num_chitta_slots)]
                    if self.config.disable_c_read_in_b:
                        c_slots = torch.zeros_like(c_slots)
                    # Build context and injection (inject only into B tokens; zeros elsewhere)
                    context = torch.cat([z_B, z_M, c_slots], dim=-2)
                    # Prepare B injection matching B length
                    c_read = self._c_read_summary(z_C).unsqueeze(1).to(z_M.dtype)
                    if self.config.disable_c_read_in_b:
                        c_read = torch.zeros_like(c_read)
                    inj_B_main = z_M + c_read  # [batch, base_len, hidden]
                    if plan_tokens > 0:
                        zero_plans = torch.zeros(z_B.shape[0], plan_tokens, z_B.shape[-1], dtype=z_B.dtype, device=z_B.device)
                        inj_B_tokens = torch.cat([zero_plans, inj_B_main], dim=-2)
                    else:
                        inj_B_tokens = inj_B_main
                    if self.config.use_film:
                        inj_B_tokens = self._apply_film(inj_B_tokens, alpha_B, beta_B)
                    inj_context = torch.cat([inj_B_tokens,
                                             torch.zeros_like(z_M),
                                             torch.zeros_like(c_slots)], dim=-2)
                    # Run B layers over augmented context, then slice back B tokens
                    hs = context + inj_context
                    for layer in self.B_level.layers:
                        hs = layer(hidden_states=hs, **seq_info_for(hs.shape[-2]))
                    z_B = hs[:, :z_B.shape[-2]]

                # A updates from B using A-shaped injection with [EGO] pooling
                base_len = input_embeddings.shape[-2]
                inj_A = torch.zeros_like(z_A)
                # [EGO] token receives pooled B
                inj_A[:, 0] = self._mean_pool(z_B)
                # Remaining A tokens align to last base_len tokens of B (exclude plan tokens)
                inj_A[:, 1:] = z_B[:, -base_len:]
                z_A = self.A_level(z_A, inj_A, **seq_info_for(z_A.shape[-2]))

                # C slow write (skip last A step before grad)
                is_last_A = (_a_step == self.config.A_cycles - 1)
                if (not self.config.freeze_c_writes) and (((_a_step + 1) % max(1, self.config.C_every_A) == 0) and not is_last_A):
                    z_C = self._c_write_update(z_C, z_A, z_B, z_M, g_C_write)

        assert not z_C.requires_grad and not z_M.requires_grad and not z_B.requires_grad and not z_A.requires_grad

        # One grad-enabled macro step (M->B->A, optional C write)
        alpha_M, beta_M, alpha_B, beta_B, g_C_write = self._film_from_A(z_A)

        inj_M = input_embeddings
        if self.config.use_film:
            inj_M = self._apply_film(inj_M, alpha_M, beta_M)
        z_M = self.M_level(z_M, inj_M, **seq_info_for(z_M.shape[-2]))

        # B grad step with context
        plan_tokens = max(0, self.config.num_plan_tokens)
        c_slots = z_C[:, :max(1, self.config.num_chitta_slots)]
        if self.config.disable_c_read_in_b:
            c_slots = torch.zeros_like(c_slots)
        context = torch.cat([z_B, z_M, c_slots], dim=-2)
        c_read = self._c_read_summary(z_C).unsqueeze(1).to(z_M.dtype)
        if self.config.disable_c_read_in_b:
            c_read = torch.zeros_like(c_read)
        inj_B_main = z_M + c_read
        if plan_tokens > 0:
            zero_plans = torch.zeros(z_B.shape[0], plan_tokens, z_B.shape[-1], dtype=z_B.dtype, device=z_B.device)
            inj_B_tokens = torch.cat([zero_plans, inj_B_main], dim=-2)
        else:
            inj_B_tokens = inj_B_main
        if self.config.use_film:
            inj_B_tokens = self._apply_film(inj_B_tokens, alpha_B, beta_B)
        inj_context = torch.cat([inj_B_tokens,
                                 torch.zeros_like(z_M),
                                 torch.zeros_like(c_slots)], dim=-2)
        hs = context + inj_context
        for layer in self.B_level.layers:
            hs = layer(hidden_states=hs, **seq_info_for(hs.shape[-2]))
        z_B = hs[:, :z_B.shape[-2]]

        # A grad update from B with A-shaped injection
        base_len = input_embeddings.shape[-2]
        inj_A = torch.zeros_like(z_A)
        inj_A[:, 0] = self._mean_pool(z_B)
        inj_A[:, 1:] = z_B[:, -base_len:]
        z_A = self.A_level(z_A, inj_A, **seq_info_for(z_A.shape[-2]))

        # Optional C write on grad step as well
        if (not self.config.freeze_c_writes) and (max(1, self.config.C_every_A) == 1):
            z_C = self._c_write_update(z_C, z_A, z_B, z_M, g_C_write)

        # Expose last g_C_write for regularization in outer wrapper
        self._last_g_C_write = g_C_write
        self._last_alpha_M = alpha_M
        self._last_alpha_B = alpha_B

        # LM/Q heads from A
        new_carry = HierarchicalReasoningModel4L_ACTV2InnerCarry(z_C=z_C.detach(), z_M=z_M.detach(), z_B=z_B.detach(), z_A=z_A.detach())
        # Exclude puzzle prefix and [EGO] token from LM outputs
        ego_offset = 1 if self.config.ego_token else 0
        output = self.lm_head(z_A)[:, self.puzzle_emb_len + ego_offset:]

        # Q-head reads from [EGO] token (index 0 if enabled, else 0 anyway)
        q_index = 0
        q_logits = self.q_head(z_A[:, q_index]).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel4L_ACTV2(nn.Module):
    """ACT wrapper for 4-level HRM v2 (Vedic-rooted)."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel4L_ACTV2Config(**config_dict)
        self.inner = HierarchicalReasoningModel4L_ACTV2_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return HierarchicalReasoningModel4L_ACTV2Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, will be reset in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: HierarchicalReasoningModel4L_ACTV2Carry, batch: Dict[str, torch.Tensor]):
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q (bootstrapping)
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                
                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        # Surface last g_C_write for optional regularization in training loop
        if getattr(self.inner, "_last_g_C_write", None) is not None:
            outputs["g_C_write"] = self.inner._last_g_C_write
            
        # Surface FiLM alphas for debugging if available
        if getattr(self.inner, "_last_alpha_M", None) is not None:
            outputs["alpha_M_mean"] = self.inner._last_alpha_M.mean().item()
            outputs["alpha_B_mean"] = self.inner._last_alpha_B.mean().item()

        return HierarchicalReasoningModel4L_ACTV2Carry(new_inner_carry, new_steps, halted, new_current_data), outputs



