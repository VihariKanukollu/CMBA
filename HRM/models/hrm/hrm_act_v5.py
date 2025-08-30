from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention as BaseAttention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class HierarchicalReasoningModel_ACTV5InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV5Carry:
    inner_carry: HierarchicalReasoningModel_ACTV5InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV5Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    # New: grouped KV heads
    n_kv_heads: int = 0  # 0 => defaults to num_heads (no GQA)
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    # RoPE / long context
    rope_theta: float = 500000.0  # high-theta like Llama 3
    original_seq_len: int = 2048
    rope_factor: float = 1.0  # >1 to extend; simple pass-through placeholder
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0

    # ACT / halting
    halt_max_steps: int
    halt_exploration_prob: float
    step_penalty: float = 0.0  # encourage early halting
    ema_target_momentum: float = 0.0  # 0 disables EMA target smoothing

    # Forward type
    forward_dtype: str = "bfloat16"

    # Optional small answer head for numeric tokens (e.g., 0-9, ops)
    answer_vocab_size: int = 0  # 0 disables


class GQAAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, n_kv_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads > 0 else num_heads
        assert self.num_heads % self.n_kv_heads == 0
        self.n_rep = self.num_heads // self.n_kv_heads

        head_dim = hidden_size // num_heads
        # Use base CastedLinear; projection shapes mirror Llama3-style split
        self.wq = CastedLinear(hidden_size, num_heads * head_dim, bias=False)
        self.wk = CastedLinear(hidden_size, self.n_kv_heads * head_dim, bias=False)
        self.wv = CastedLinear(hidden_size, self.n_kv_heads * head_dim, bias=False)
        self.wo = CastedLinear(num_heads * head_dim, hidden_size, bias=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        # x: [bs, seq, n_kv_heads, head_dim] -> [bs, seq, n_heads, head_dim]
        if self.n_rep == 1:
            return x
        bs, seqlen, n_kv, hd = x.shape
        return x[:, :, :, None, :].expand(bs, seqlen, n_kv, self.n_rep, hd).reshape(bs, seqlen, n_kv * self.n_rep, hd)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        bs, seqlen, _ = hidden_states.shape
        head_dim = self.hidden_size // self.num_heads

        q = self.wq(hidden_states).view(bs, seqlen, self.num_heads, head_dim)
        k = self.wk(hidden_states).view(bs, seqlen, self.n_kv_heads, head_dim)
        v = self.wv(hidden_states).view(bs, seqlen, self.n_kv_heads, head_dim)

        # Apply RoPE if provided
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = BaseAttention.apply_rotary_pos_emb(q, k, cos, sin) if hasattr(BaseAttention, 'apply_rotary_pos_emb') else (q, k)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Flash attention path exists in Base Attention; here fallback to manual
        q = q.transpose(1, 2)  # [bs, heads, seq, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        out = torch.matmul(scores, v)  # [bs, heads, seq, hd]
        out = out.transpose(1, 2).contiguous().view(bs, seqlen, -1)
        return self.wo(out)


class HierarchicalReasoningModel_ACTV5Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV5Config):
        super().__init__()

        head_dim = config.hidden_size // config.num_heads
        n_kv = config.n_kv_heads if config.n_kv_heads > 0 else config.num_heads

        if n_kv != config.num_heads:
            self.self_attn = GQAAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                n_kv_heads=n_kv,
            )
        else:
            self.self_attn = BaseAttention(
                hidden_size=config.hidden_size,
                head_dim=head_dim,
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
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_ACTV5ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV5Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class HierarchicalReasoningModel_ACTV5_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV5Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)
        self.answer_head  = None
        if self.config.answer_vocab_size and self.config.answer_vocab_size > 0:
            self.answer_head = CastedLinear(self.config.hidden_size, self.config.answer_vocab_size, bias=False)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # Positional encodings
        if self.config.pos_encodings == "rope":
            # Use high-theta RoPE by passing larger base
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV5ReasoningModule(layers=[HierarchicalReasoningModel_ACTV5Block(self.config) for _i in range(self.config.H_layers)])
        self.L_level = HierarchicalReasoningModel_ACTV5ReasoningModule(layers=[HierarchicalReasoningModel_ACTV5Block(self.config) for _i in range(self.config.L_layers)])
        
        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

        # EMA buffer for target smoothing if enabled
        if self.config.ema_target_momentum and self.config.ema_target_momentum > 0:
            self.register_buffer("_ema_q_continue", torch.zeros((), dtype=torch.float32), persistent=False)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV5InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV5InnerCarry):
        return HierarchicalReasoningModel_ACTV5InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV5InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV5InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L
            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # Outputs
        new_carry = HierarchicalReasoningModel_ACTV5InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        answer_logits = None
        if self.answer_head is not None:
            answer_logits = self.answer_head(z_H[:, -1])  # use last position as compact answer hint
        
        return new_carry, logits, (q_logits[..., 0], q_logits[..., 1]), answer_logits


class HierarchicalReasoningModel_ACTV5(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV5Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV5_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return HierarchicalReasoningModel_ACTV5Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: HierarchicalReasoningModel_ACTV5Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV5Carry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        new_inner_carry, logits, (q_halt_logits, q_continue_logits), answer_logits = self.inner(new_inner_carry, new_current_data)

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        if answer_logits is not None:
            outputs["answer_logits"] = answer_logits
        
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                # Target Q smoothing (EMA) if enabled
                next_q_halt, next_q_cont = self.inner(new_inner_carry, new_current_data)[-2]
                target = torch.sigmoid(torch.where(is_last_step, next_q_halt, torch.maximum(next_q_halt, next_q_cont)))
                if self.config.ema_target_momentum and self.config.ema_target_momentum > 0:
                    m = float(self.config.ema_target_momentum)
                    if not hasattr(self, "_ema_buf"):
                        self._ema_buf = target.detach().mean()  # type: ignore[attr-defined]
                    self._ema_buf = (1 - m) * self._ema_buf + m * target.detach().mean()  # type: ignore[attr-defined]
                    target = 0.5 * target + 0.5 * self._ema_buf  # type: ignore[attr-defined]
                outputs["target_q_continue"] = target

            # Optional step penalty via metric for loss head to consume
            if self.config.step_penalty and self.config.step_penalty > 0:
                outputs["step_penalty"] = (new_steps.to(torch.float32) * float(self.config.step_penalty))

        return HierarchicalReasoningModel_ACTV5Carry(new_inner_carry, new_steps, halted, new_current_data), outputs


