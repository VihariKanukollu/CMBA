from typing import Tuple, List, Dict, Optional
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
class HierarchicalReasoningModel6L_ACTV3InnerCarry:
    z_levels: List[torch.Tensor]


@dataclass
class HierarchicalReasoningModel6L_ACTV3Carry:
    inner_carry: HierarchicalReasoningModel6L_ACTV3InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel6L_ACTV3Config(BaseModel):
    # Input/config
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    # N-level schedule (fast -> slow)
    num_levels: int = 6
    levels_layers: Optional[List[int]] = None
    levels_cycles: Optional[List[int]] = None

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Injection/gating
    input_injection: str = "gated_add"  # "add" | "gated_add" | "cross_attn"
    topdown_to_bottom: bool = True
    cross_attn_levels: Optional[List[int]] = None  # which levels use cross-attn integration
    cross_attn_in_settle: bool = False  # apply cross-attn during no_grad settle

    # Graph reasoning (for Level II/III)
    enable_graph_reasoning: bool = False
    graph_structure: str = "learned"  # "learned" | "grid" | "fully_connected"
    graph_top_k: int = 8               # sparsify adjacency to top-k neighbors (<=0 disables)
    graph_in_settle: bool = False      # compute graph msgs during no_grad settle

    # Consistency-driven halting bias (eval-only by default)
    consistency_halting_bias: float = 0.0
    consistency_bias_eval_only: bool = True

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"


class HierarchicalReasoningModel_ACTV3Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel6L_ACTV3Config) -> None:
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


class HierarchicalReasoningModel_ACTV3ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV3Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Optional cross-attention adapter (honor settle toggle)
        if "cross_attn_adapter" in kwargs and kwargs["cross_attn_adapter"] is not None and ("global_context" in kwargs):
            adapter: CrossAttentionAdapter = kwargs["cross_attn_adapter"]
            global_context: torch.Tensor = kwargs["global_context"]
            cfg = getattr(self, "config", None)
            if hidden_states.requires_grad or (cfg is not None and getattr(cfg, "cross_attn_in_settle", False)):
                input_injection = input_injection + adapter(hidden_states, global_context)

        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class DilatedTemporalConvolutionModule(nn.Module):
    """Temporal convolution stack with exponentially increasing dilation.
    Matches the common signature and supports optional cross-attn via kwargs.
    """

    def __init__(self, config: HierarchicalReasoningModel6L_ACTV3Config, num_layers: int):
        super().__init__()
        self.config = config
        self.norm_eps = config.rms_norm_eps
        hidden = config.hidden_size

        blocks = []
        for i in range(max(1, num_layers)):
            dilation = 2 ** i
            pad = dilation  # padding to keep length invariant with kernel_size=3
            conv = nn.Conv1d(hidden, hidden, kernel_size=3, dilation=dilation, padding=pad, groups=1, bias=False)
            # Ensure convolution runs in forward dtype
            conv = conv.to(getattr(torch, self.config.forward_dtype))
            proj = CastedLinear(hidden, hidden, bias=False)
            blocks.append(nn.ModuleDict({
                "conv": conv,
                "proj": proj,
                "mlp": SwiGLU(hidden_size=hidden, expansion=self.config.expansion)
            }))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Optional cross attention enrichment of injection (honor settle toggle)
        if "cross_attn_adapter" in kwargs and kwargs["cross_attn_adapter"] is not None and ("global_context" in kwargs):
            adapter: CrossAttentionAdapter = kwargs["cross_attn_adapter"]
            global_context: torch.Tensor = kwargs["global_context"]
            cfg = getattr(self, "config", None)
            if hidden_states.requires_grad or (cfg is not None and getattr(cfg, "cross_attn_in_settle", False)):
                input_injection = input_injection + adapter(hidden_states, global_context)

        x = hidden_states + input_injection
        # Conv expects [bs, hidden, seq]
        x_ch = x.transpose(1, 2)
        for blk in self.blocks:
            y = blk["conv"](x_ch)
            # remove extra padding effect by trimming to original length
            if y.shape[-1] != x_ch.shape[-1]:
                y = y[..., : x_ch.shape[-1]]
            y = y.transpose(1, 2)
            y = rms_norm(x + blk["proj"](y), variance_epsilon=self.norm_eps)
            x = rms_norm(y + blk["mlp"](y), variance_epsilon=self.norm_eps)
            x_ch = x.transpose(1, 2)
        return x


class ReservoirGraphReasoningModule(nn.Module):
    """Fast reservoir-like update with graph message passing.
    If graph is provided via kwargs["graph"], use it to aggregate messages.
    Supports optional cross-attn integration via kwargs.
    """

    def __init__(self, config: HierarchicalReasoningModel6L_ACTV3Config, num_layers: int):
        super().__init__()
        self.config = config
        self.norm_eps = config.rms_norm_eps
        hidden = config.hidden_size

        # Simple learned message and state mixing
        self.in_proj = CastedLinear(hidden, hidden, bias=False)
        self.msg_proj = CastedLinear(hidden, hidden, bias=False)
        self.mlp = SwiGLU(hidden_size=hidden, expansion=config.expansion)
        self.layers = max(1, num_layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        graph = kwargs.get("graph", None)
        # Optional cross-attn for richer input (honor settle toggle)
        if "cross_attn_adapter" in kwargs and kwargs["cross_attn_adapter"] is not None and ("global_context" in kwargs):
            adapter: CrossAttentionAdapter = kwargs["cross_attn_adapter"]
            global_context: torch.Tensor = kwargs["global_context"]
            cfg = getattr(self, "config", None)
            if hidden_states.requires_grad or (cfg is not None and getattr(cfg, "cross_attn_in_settle", False)):
                input_injection = input_injection + adapter(hidden_states, global_context)

        x = hidden_states
        for _ in range(self.layers):
            inj = self.in_proj(input_injection)
            if graph is not None:
                # Use graph only during grad pass unless enabled for settle
                cfg = getattr(self, "config", None)
                use_graph = hidden_states.requires_grad or (cfg is not None and getattr(cfg, "graph_in_settle", False))
                if use_graph:
                    # graph: [bs, seq, seq]; x: [bs, seq, hidden]
                    top_k = (getattr(cfg, "graph_top_k", 0) if cfg is not None else 0)
                    if isinstance(top_k, int) and top_k > 0 and top_k < graph.shape[-1]:
                        values, indices = torch.topk(graph, k=top_k, dim=-1)
                        neighbors = torch.gather(
                            x.unsqueeze(1).expand(-1, x.shape[-2], -1, -1),
                            2,
                            indices.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1])
                        )
                        msg = (values.unsqueeze(-1).to(x.dtype) * neighbors).sum(dim=2)
                    else:
                        msg = torch.matmul(graph.to(x.dtype), x)
                    msg = self.msg_proj(msg)
                else:
                    msg = 0.0
            else:
                msg = 0.0
            x = rms_norm(x + inj + msg, variance_epsilon=self.norm_eps)
            x = rms_norm(x + self.mlp(x), variance_epsilon=self.norm_eps)
        return x

class CrossAttentionAdapter(nn.Module):
    """Simple cross-attention adapter used for attention-based integration at the fastest level.
    Q projects from hidden_states; K,V project from context_states (e.g., inputs or top-down context).
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.k_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.v_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.o_proj = CastedLinear(hidden_size, hidden_size, bias=False)

        # Near-identity init for stability (small projections)
        with torch.no_grad():
            for p in [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.o_proj.weight]:
                p.mul_(0.02)

    def forward(self, hidden_states: torch.Tensor, context_states: torch.Tensor) -> torch.Tensor:
        bs, tgt_len, _ = hidden_states.shape
        _, src_len, _ = context_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(context_states)
        v = self.v_proj(context_states)

        # [bs, seq, heads, head_dim] -> [bs, heads, seq, head_dim]
        def reshape(x, seqlen):
            return x.view(bs, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape(q, tgt_len)
        k = reshape(k, src_len)
        v = reshape(v, src_len)

        # Scaled dot product attention (uses FlashAttention kernel when available in PyTorch)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        # [bs, heads, tgt_len, head_dim] -> [bs, tgt_len, hidden]
        attn = attn.transpose(1, 2).contiguous().view(bs, tgt_len, self.hidden_size)
        return self.o_proj(attn)


class HierarchicalReasoningModel6L_ACTV3_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel6L_ACTV3Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # Defaults for layers/cycles if not provided
        if self.config.levels_layers is None:
            self.config.levels_layers = [2, 2, 2, 2, 2, 2][: self.config.num_levels]
        if self.config.levels_cycles is None:
            # Fast -> slow cycles (illustrative defaults)
            self.config.levels_cycles = [8, 4, 2, 1, 1, 1][: self.config.num_levels]

        assert len(self.config.levels_layers) == self.config.num_levels
        assert len(self.config.levels_cycles) == self.config.num_levels

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
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        # Reasoning layers per level (specialized factory)
        self.levels = torch.nn.ModuleList([
            self._create_cortical_level(i) for i in range(self.config.num_levels)
        ])

        # Optional gating per level (FiLM-like), driven by top-level pooled state
        self.gates = torch.nn.ModuleList([
            nn.Linear(self.config.hidden_size, 2 * self.config.hidden_size, bias=True)
            for _ in range(self.config.num_levels)
        ])
        with torch.no_grad():
            for g in self.gates:
                g.weight.zero_()
                g.bias.zero_()

        # Cross-attn adapters for selected levels
        self.cross_attn_adapters = None
        if self.config.input_injection == "cross_attn":
            levels = self.config.cross_attn_levels if (self.config.cross_attn_levels is not None) else [0]
            mask = [i in levels for i in range(self.config.num_levels)]
            self.cross_attn_adapters = nn.ModuleList([
                CrossAttentionAdapter(self.config.hidden_size, self.config.num_heads) if use else None  # type: ignore
                for use in mask
            ])

        # Initial states (one vector per level, broadcast at reset)
        # Use registered buffers instead of ModuleList to avoid container type constraints
        self.level_init_names: List[str] = []
        for i in range(self.config.num_levels):
            buf = trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1)
            name = f"level_init_{i}"
            self.register_buffer(name, buf, persistent=True)
            self.level_init_names.append(name)

        # Q head special init (near zero)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _create_cortical_level(self, level_idx: int) -> nn.Module:
        num_layers = self.config.levels_layers[level_idx]
        # Mapping: 0 -> Temporal conv (Layer IV), 1 -> Reservoir+Graph (Layer II/III),
        # 2..(K-2) -> Transformer reasoning, (K-1) -> Top/error-monitoring as transformer block stack
        if level_idx == 0:
            return DilatedTemporalConvolutionModule(self.config, num_layers=num_layers)
        if level_idx == 1 and self.config.enable_graph_reasoning:
            return ReservoirGraphReasoningModule(self.config, num_layers=num_layers)
        # Default transformer stack
        return HierarchicalReasoningModel_ACTV3ReasoningModule(
            layers=[HierarchicalReasoningModel_ACTV3Block(self.config) for _ in range(num_layers)]
        )

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
        device = next(self.parameters()).device
        base_len = self.config.seq_len + self.puzzle_emb_len
        return HierarchicalReasoningModel6L_ACTV3InnerCarry(
            z_levels=[
                torch.empty(batch_size, base_len, self.config.hidden_size, dtype=self.forward_dtype, device=device)
                for _ in range(self.config.num_levels)
            ]
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel6L_ACTV3InnerCarry):
        # Broadcast the init vector to the full sequence length when resetting
        def reset_one(idx: int, z: torch.Tensor):
            init_vec = getattr(self, self.level_init_names[idx])
            return torch.where(reset_flag.view(-1, 1, 1), init_vec, z)

        return HierarchicalReasoningModel6L_ACTV3InnerCarry(
            z_levels=[reset_one(i, z) for i, z in enumerate(carry.z_levels)]
        )

    @staticmethod
    def _mean_pool(hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.mean(dim=-2)

    def _apply_gate(self, level_idx: int, injection: torch.Tensor, top_state: torch.Tensor) -> torch.Tensor:
        if self.config.input_injection != "gated_add":
            return injection
        pooled = self._mean_pool(top_state).to(torch.float32)
        alpha_beta = self.gates[level_idx](pooled)
        hs = self.config.hidden_size
        alpha_raw, beta_raw = alpha_beta.split([hs, hs], dim=-1)
        alpha = 1.0 + alpha_raw
        beta = beta_raw
        d = injection.dtype
        return alpha.to(d).unsqueeze(1) * injection + beta.to(d).unsqueeze(1)

    def _compute_injection(self, idx: int, z_levels: List[torch.Tensor], input_embeddings: torch.Tensor) -> torch.Tensor:
        if idx == 0:
            inj = input_embeddings
            if self.config.topdown_to_bottom:
                inj = inj + z_levels[-1]
        else:
            inj = z_levels[idx - 1]
        # Apply gating (if enabled)
        inj = self._apply_gate(idx, inj, z_levels[-1])
        return inj

    def _build_compact_global_context(self, z_levels: List[torch.Tensor], input_embeddings: torch.Tensor) -> torch.Tensor:
        # inputs + pooled top + pooled per-level tokens -> [bs, ctx_len, hidden]
        bs, seq_len, hidden = input_embeddings.shape
        top_pool = self._mean_pool(z_levels[-1]).unsqueeze(1).to(input_embeddings.dtype)
        per_level_pools = torch.stack([self._mean_pool(z).to(input_embeddings.dtype) for z in z_levels], dim=1)
        return torch.cat([input_embeddings, top_pool, per_level_pools], dim=-2)

    def _build_puzzle_graph(self, input_embeddings: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.config.enable_graph_reasoning:
            return None
        method = self.config.graph_structure
        x = input_embeddings.to(torch.float32)
        bs, seq_len, hidden = x.shape
        if method == "fully_connected":
            adj = torch.full((bs, seq_len, seq_len), 1.0 / float(seq_len), device=x.device, dtype=x.dtype)
            return adj.to(input_embeddings.dtype)
        if method == "learned":
            # Softmax over similarity (scaled dot-product)
            sim = torch.matmul(x, x.transpose(1, 2)) / max(1.0, math.sqrt(hidden))
            adj = torch.softmax(sim, dim=-1)
            return adj.to(input_embeddings.dtype)
        if method == "grid":
            # Fallback: use a banded adjacency (local 1-hop) if true grid unknown
            eye = torch.eye(seq_len, device=x.device, dtype=x.dtype)
            band = eye + torch.roll(eye, shifts=1, dims=0) + torch.roll(eye, shifts=-1, dims=0)
            band = band.unsqueeze(0).expand(bs, -1, -1)
            band = band / band.sum(dim=-1, keepdim=True).clamp_min(1)
            return band.to(input_embeddings.dtype)
        return None

    def forward(self, carry: HierarchicalReasoningModel6L_ACTV3InnerCarry, batch: Dict[str, torch.Tensor]):
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        K = self.config.num_levels
        levels_cycles = self.config.levels_cycles

        # Build per-level kwargs (cross-attn and graph) once per forward
        level_kwargs: Dict[int, Dict[str, torch.Tensor]] = {}
        if self.cross_attn_adapters is not None:
            global_context = self._build_compact_global_context(carry.z_levels, input_embeddings)
            for i, adapter in enumerate(self.cross_attn_adapters):
                if adapter is not None:
                    level_kwargs.setdefault(i, {})
                    level_kwargs[i]["global_context"] = global_context
                    level_kwargs[i]["cross_attn_adapter"] = adapter

        graph = self._build_puzzle_graph(input_embeddings)
        if graph is not None:
            # Assume level 1 consumes graph
            level_kwargs.setdefault(1, {})
            level_kwargs[1]["graph"] = graph

        # Forward iterations (settle without grad, skipping the global last combination)
        with torch.no_grad():
            z_levels = list(carry.z_levels)

            def run_no_grad_level(i: int, is_final_path: bool):
                cycles = levels_cycles[i]
                for step in range(cycles):
                    last_here = (step == cycles - 1)
                    child_final = is_final_path and last_here
                    if i > 0:
                        run_no_grad_level(i - 1, child_final)
                    # Skip this level update only on the global final combination
                    if not child_final:
                        inj = self._compute_injection(i, z_levels, input_embeddings)
                        z_levels[i] = self.levels[i](hidden_states=z_levels[i], input_injection=inj, **seq_info, **level_kwargs.get(i, {}))

            # Start from top level, marking the whole path as final for the last combination
            run_no_grad_level(K - 1, True)

        # One grad-enabled bottom-to-top sweep (1-step grad)
        assert all(not z.requires_grad for z in carry.z_levels)
        z_levels = list(carry.z_levels)

        for i in range(K):
            inj = self._compute_injection(i, z_levels, input_embeddings)
            z_levels[i] = self.levels[i](hidden_states=z_levels[i], input_injection=inj, **seq_info, **level_kwargs.get(i, {}))

        # LM/Q heads from top level
        new_carry = HierarchicalReasoningModel6L_ACTV3InnerCarry(z_levels=[z.detach() for z in z_levels])
        output = self.lm_head(z_levels[-1])[:, self.puzzle_emb_len:]

        q_logits = self.q_head(z_levels[-1][:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel4L_ACTV3(nn.Module):
    """ACT wrapper for N-level HRM v3 (default 6 levels).

    The class name is kept for compatibility with loader identifiers; the actual number of levels is configured via the config.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel6L_ACTV3Config(**config_dict)
        self.inner = HierarchicalReasoningModel6L_ACTV3_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return HierarchicalReasoningModel6L_ACTV3Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reset in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: HierarchicalReasoningModel6L_ACTV3Carry, batch: Dict[str, torch.Tensor]):
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
        
        # Consistency score across adjacent levels (aux metric only)
        with torch.no_grad():
            zls = new_inner_carry.z_levels
            if isinstance(zls, list) and len(zls) >= 2:
                sims = []
                for i in range(1, len(zls)):
                    a = zls[i].mean(dim=-2).to(torch.float32)
                    b = zls[i - 1].mean(dim=-2).to(torch.float32)
                    sim = F.cosine_similarity(a, b, dim=-1)
                    sims.append(sim)
                if len(sims):
                    outputs["consistency_score"] = torch.stack(sims, dim=0).mean(dim=0)
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                if (not self.config.consistency_bias_eval_only) and ("consistency_score" in outputs) and (self.config.consistency_halting_bias != 0.0):
                    cprob = torch.clamp((outputs["consistency_score"] + 1.0) * 0.5, 0.0, 1.0)
                    biased_halt = q_halt_logits + self.config.consistency_halting_bias * (1.0 - cprob)
                    biased_cont = q_continue_logits + self.config.consistency_halting_bias * cprob
                    halted = halted | (biased_halt > biased_cont)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q (bootstrapping)
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                
                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

            # Inference-time bias (eval only)
            if (not self.training) and (self.config.halt_max_steps > 1):
                if ("consistency_score" in outputs) and (self.config.consistency_halting_bias != 0.0):
                    cprob = torch.clamp((outputs["consistency_score"] + 1.0) * 0.5, 0.0, 1.0)
                    biased_halt = q_halt_logits + self.config.consistency_halting_bias * (1.0 - cprob)
                    biased_cont = q_continue_logits + self.config.consistency_halting_bias * cprob
                    halted = halted | (biased_halt > biased_cont)

        return HierarchicalReasoningModel6L_ACTV3Carry(new_inner_carry, new_steps, halted, new_current_data), outputs


# Optional alias to a more explicit 6L class name for convenience
class HierarchicalReasoningModel6L_ACTV3(HierarchicalReasoningModel4L_ACTV3):
    pass


