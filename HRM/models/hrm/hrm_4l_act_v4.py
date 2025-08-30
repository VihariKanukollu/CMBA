from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import torch.distributed as dist

from models.common import trunc_normal_init_
from models.layers import SwiGLU, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


world_size = 1
rank = 0


class RMSNormLearnable(nn.Module):
    """Learnable RMSNorm (Pre-Norm usage).

    Matches DeepSeek-style RMSNorm with learnable scale.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class YarnRotaryEmbedding(nn.Module):
    """Yarn-extended RoPE for long context stability.

    Adapts rotary frequencies beyond original_seq_len using rope_factor and beta params.
    Produces (cos, sin) like the existing RotaryEmbedding.
    """
    def __init__(self, dim: int, max_position_embeddings: int, base: float,
                 original_seq_len: int, rope_factor: float, beta_fast: int, beta_slow: int, device=None,
                 mscale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.original_seq_len = original_seq_len
        self.rope_factor = rope_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

        if max_position_embeddings > original_seq_len:
            def find_correction_dim(num_rotations, d, b, max_seq_len):
                return d * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(b))

            def find_correction_range(low_rot, high_rot, d, b, max_seq_len):
                low = math.floor(find_correction_dim(low_rot, d, b, max_seq_len))
                high = math.ceil(find_correction_dim(high_rot, d, b, max_seq_len))
                return max(low, 0), min(high, d - 1)

            def linear_ramp_factor(vmin, vmax, d):
                if vmin == vmax:
                    vmax += 1e-3
                linear_func = (torch.arange(d, dtype=torch.float32, device=device) - vmin) / (vmax - vmin)
                return torch.clamp(linear_func, 0, 1)

            low, high = find_correction_range(self.beta_fast, self.beta_slow, dim, base, original_seq_len)
            smooth = 1 - linear_ramp_factor(low, high, dim // 2)
            inv_freq = inv_freq / rope_factor * (1 - smooth) + inv_freq * smooth

        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self) -> CosSin:
        return self.cos_cached, self.sin_cached


def _find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b


class ColumnParallelLinearV3(nn.Module):
    """Column Parallel Linear with real sharding if torch.distributed is initialized."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        assert out_features % max(world_size, 1) == 0
        self.part_out = out_features // max(world_size, 1)
        self.bias = nn.Parameter(torch.zeros(self.part_out)) if bias else None
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty((self.part_out, in_features), dtype=torch.float32), std=1.0 / (in_features ** 0.5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight.to(x.dtype), bias=self.bias.to(x.dtype) if self.bias is not None else None)
        return y


class RowParallelLinearV3(nn.Module):
    """Row Parallel Linear with real sharding if torch.distributed is initialized."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        assert in_features % max(world_size, 1) == 0
        self.part_in = in_features // max(world_size, 1)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty((out_features, self.part_in), dtype=torch.float32), std=1.0 / (self.part_in ** 0.5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features] -> split along last dim
        if world_size > 1:
            x_part = torch.narrow(x, dim=-1, start=rank * self.part_in, length=self.part_in)
        else:
            x_part = x
        y_part = F.linear(x_part, self.weight.to(x.dtype), bias=None)
        if world_size > 1:
            dist.all_reduce(y_part)
        if self.bias is not None:
            y_part = y_part + self.bias.to(y_part.dtype)
        return y_part


class ParallelEmbeddingV3(nn.Module):
    def __init__(self, vocab_size: int, dim: int, cast_to: torch.dtype):
        super().__init__()
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        assert vocab_size % max(world_size, 1) == 0
        self.dim = dim
        self.cast_to = cast_to
        self.part_vocab_size = vocab_size // max(world_size, 1)
        self.vocab_start = rank * self.part_vocab_size
        self.vocab_end = self.vocab_start + self.part_vocab_size
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty((self.part_vocab_size, dim), dtype=torch.float32), std=1.0 / (dim ** 0.5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            mask = (x < self.vocab_start) | (x >= self.vocab_end)
            x_local = x - self.vocab_start
            x_local = x_local.masked_fill(mask, 0)
        else:
            x_local = x
            mask = None
        y_local = F.embedding(x_local, self.weight.to(self.cast_to))
        if world_size > 1:
            y_local = y_local.masked_fill(mask.unsqueeze(-1), 0)
            dist.all_reduce(y_local)
        return y_local


class GateV3(nn.Module):
    """Group-limited routing with noise/temp and route_scale."""
    def __init__(self, dim: int, n_routed_experts: int, n_activated_experts: int,
                 n_expert_groups: int = 1, n_limited_groups: int = 1,
                 route_scale: float = 1.0, router_noise_std: float = 0.0, router_temperature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.n_expert_groups = max(1, n_expert_groups)
        self.n_limited_groups = max(1, n_limited_groups)
        self.route_scale = route_scale
        self.router_noise_std = router_noise_std
        self.router_temperature = max(1e-3, router_temperature)
        self.weight = nn.Parameter(torch.empty(n_routed_experts, dim))
        nn.init.normal_(self.weight, mean=0.0, std=dim ** -0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cast gate weights to match activations dtype to avoid bf16/fp32 mm mismatch
        logits = F.linear(x, self.weight.to(x.dtype))
        if self.router_noise_std > 0:
            logits = logits + self.router_noise_std * torch.randn_like(logits)
        logits = logits / self.router_temperature
        scores = logits.softmax(dim=-1)
        original_scores = scores
        if self.n_expert_groups > 1:
            scores = scores.view(x.size(0), self.n_expert_groups, -1)
            group_scores = scores.amax(dim=-1)
            group_top = group_scores.topk(self.n_limited_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_expert_groups, dtype=bool).scatter(1, group_top, False)
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.n_activated_experts, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        weights = weights * self.route_scale
        return weights.type_as(x), indices


class ExpertV3(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = CastedLinear(dim, inter_dim, bias=False)
        self.w2 = CastedLinear(inter_dim, dim, bias=False)
        self.w3 = CastedLinear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEV3(nn.Module):
    def __init__(self, dim: int, inter_dim: int, n_routed_experts: int, n_activated_experts: int,
                 n_expert_groups: int = 1, n_limited_groups: int = 1, route_scale: float = 1.0,
                 capacity_factor: float = 1.25, router_noise_std: float = 0.0, router_temperature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.capacity_factor = capacity_factor
        self.gate = GateV3(dim, n_routed_experts, n_activated_experts,
                            n_expert_groups=n_expert_groups, n_limited_groups=n_limited_groups,
                            route_scale=route_scale, router_noise_std=router_noise_std,
                            router_temperature=router_temperature)
        self.experts = nn.ModuleList([ExpertV3(dim, inter_dim) for _ in range(n_routed_experts)])
        # Shared experts path (dense MLP always on)
        self.shared_experts = SwiGLU(hidden_size=dim, expansion=max(1.0, inter_dim * 3 / (2 * dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        num_tokens = x.size(0)
        # Capacity per expert
        cap = max(1, int(self.capacity_factor * (num_tokens * self.n_activated_experts / max(1, self.n_routed_experts))))
        overflow_mask = torch.zeros(num_tokens, dtype=torch.bool, device=x.device)
        for i in range(self.n_routed_experts):
            where_mask = (indices == i)
            if not where_mask.any():
                continue
            idx, top = torch.where(where_mask)
            if idx.numel() > cap:
                keep = idx[:cap]
                keep_top = top[:cap]
                drop = idx[cap:]
                overflow_mask[drop] = True
            else:
                keep = idx
                keep_top = top
            if keep.numel() > 0:
                y[keep] += self.experts[i](x[keep]) * weights[keep, keep_top, None]
        # Shared path for all (including overflow)
        z = self.shared_experts(x)
        y = y + z
        return y.view(shape)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)


class HRMMLA(nn.Module):
    """DeepSeek-style MLA adapted to HRM shapes.

    Supports low-rank KV, rotary-only PE split, and mscale rescaling.
    """
    def __init__(self, hidden_size: int, num_heads: int,
                 q_lora_rank: int, kv_lora_rank: int,
                 qk_nope_head_dim: int, qk_rope_head_dim: int, v_head_dim: int,
                 rope_factor: float, mscale: float, attn_impl: str = "naive"):
        super().__init__()
        global world_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.local_heads = num_heads // max(world_size, 1)
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.attn_impl = attn_impl

        if q_lora_rank == 0:
            self.wq = ColumnParallelLinearV3(hidden_size, num_heads * self.qk_head_dim)
        else:
            self.wq_a = CastedLinear(hidden_size, q_lora_rank, bias=False)
            self.q_norm = RMSNormLearnable(q_lora_rank)
            self.wq_b = ColumnParallelLinearV3(q_lora_rank, num_heads * self.qk_head_dim)

        self.wkv_a = CastedLinear(hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNormLearnable(kv_lora_rank)
        self.wkv_b = ColumnParallelLinearV3(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim))
        self.wo = RowParallelLinearV3(num_heads * v_head_dim, hidden_size)

        # softmax rescale for long context
        self.softmax_scale = (self.qk_head_dim ** -0.5) * (1.0 if rope_factor <= 1.0 else (0.1 * mscale * math.log(max(rope_factor, 1.0001)) + 1.0) ** 2)
        # factor relative to default 1/sqrt(d)
        self.extra_scale = float(self.softmax_scale / (self.qk_head_dim ** -0.5))

    def _apply_rotary(self, q_pe: torch.Tensor, k_pe: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q_pe: [b, s, h, d], k_pe: [b, s, 1, d], cos/sin: [s, d]
        # Match cos/sin dtype to q_pe to avoid unintended upcasting
        cos = cos.to(q_pe.dtype)
        sin = sin.to(q_pe.dtype)
        q = (q_pe * cos.unsqueeze(-2)) + (_rotate_half(q_pe) * sin.unsqueeze(-2))
        k = (k_pe * cos.unsqueeze(-2)) + (_rotate_half(k_pe) * sin.unsqueeze(-2))
        return q, k

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.size()
        if self.q_lora_rank == 0:
            q = self.wq(hidden_states)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(hidden_states)))
        q = q.view(bsz, seqlen, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv = self.wkv_a(hidden_states)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # project kv low-rank to keys(nope) and values
        kv_proj = self.wkv_b(self.kv_norm(kv))
        kv_proj = kv_proj.view(bsz, seqlen, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        if cos_sin is not None:
            cos, sin = cos_sin
            # expand positional k_pe across heads
            k_pe = k_pe.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
            q_pe, k_pe = self._apply_rotary(q_pe, k_pe, cos, sin)

        # build full K from nope and positional
        k = torch.cat([k_nope, k_pe], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)

        if self.attn_impl == "sdp":
            # Scaled dot-product attention (flash/mem-efficient when available)
            # Adjust q by extra scaling to match custom softmax scale
            if self.extra_scale != 1.0:
                q = q * self.extra_scale
            # Permute to [b, h, s, d]
            qh = q.permute(0, 2, 1, 3)
            kh = k.permute(0, 2, 1, 3)
            vh = v.permute(0, 2, 1, 3)
            # Ensure matching dtypes across q/k/v
            if vh.dtype != qh.dtype:
                vh = vh.to(qh.dtype)
            # attn_mask shape [s, t] or broadcastable; ensure dtype matches query
            mask = attn_mask.to(qh.dtype) if attn_mask is not None else None
            # Let PyTorch pick the best kernel; avoid context-manager to keep Dynamo happy
            xh = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, dropout_p=0.0, is_causal=False)
            x = xh.permute(0, 2, 1, 3)
            x = self.wo(x.flatten(2))
            return x
        else:
            scores = torch.einsum("bshd,bthd->bsht", q, k) * self.softmax_scale
            if attn_mask is not None:
                scores = scores + attn_mask.to(scores.dtype).unsqueeze(0).unsqueeze(2)
            # Softmax in float32 for numerical stability, then cast back
            attn = scores.softmax(dim=-1, dtype=torch.float32).type_as(hidden_states)
            x = torch.einsum("bsht,bthd->bshd", attn, v)
            x = self.wo(x.flatten(2))
            return x


@dataclass
class HierarchicalReasoningModel4L_ACTV3InnerCarry:
    z_C: torch.Tensor
    z_M: torch.Tensor
    z_B: torch.Tensor
    z_A: torch.Tensor


@dataclass
class HierarchicalReasoningModel4L_ACTV3Carry:
    inner_carry: HierarchicalReasoningModel4L_ACTV3InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel4L_ACTV3Config(BaseModel):
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

    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    # Yarn extension params (optional)
    original_seq_len: Optional[int] = None
    rope_factor: float = 1.0
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0

    # MLA attention params
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    attn_impl: str = "naive"  # placeholder for future variants
    use_parallel_embedding: bool = False

    # Output vocabulary (can be much smaller than input vocab for classification-style outputs)
    output_vocab_size: int = 16
    # Auxiliary language reconstruction from M-level (input BPE recon)
    aux_recon_loss_weight: float = 0.0

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

    # MoE options
    use_moe: bool = False
    n_routed_experts: int = 8
    n_activated_experts: int = 2
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    route_scale: float = 1
    expert_capacity_factor: float = 1.25
    router_noise_std: float = 0.0
    router_temperature: float = 1.0

    # Multi-Token Prediction
    mtp_num_future: int = 0  # 0 disables MTP
    mtp_gamma: float = 0.5

    # Compile
    compile_modules: bool = False

    # Causal decoding over A-level (LM-like behavior for output region)
    causal_in_a: bool = False


class HierarchicalReasoningModel4L_ACTV3Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel4L_ACTV3Config) -> None:
        super().__init__()

        self.self_attn = HRMMLA(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            rope_factor=config.rope_factor,
            mscale=config.mscale,
            attn_impl=config.attn_impl,
        )

        inter = _find_multiple(round(config.expansion * config.hidden_size * 2 / 3), 256)
        if config.use_moe:
            self.ffn = MoEV3(dim=config.hidden_size, inter_dim=inter,
                             n_routed_experts=config.n_routed_experts,
                             n_activated_experts=config.n_activated_experts)
        else:
            self.ffn = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)

        self.attn_norm = RMSNormLearnable(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNormLearnable(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-Norm residual
        hidden_states = hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=self.attn_norm(hidden_states), attn_mask=attn_mask)
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states


class HierarchicalReasoningModel4L_ACTV3ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel4L_ACTV3Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, attn_mask=attn_mask, **kwargs)

        return hidden_states


class HierarchicalReasoningModel4L_ACTV3_Inner(nn.Module):
    """V3 inner model: Pre-Norm, optional MoE, Yarn RoPE, MTP support."""

    def __init__(self, config: HierarchicalReasoningModel4L_ACTV3Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self._last_g_C_write: Optional[torch.Tensor] = None
        self._last_alpha_M: Optional[torch.Tensor] = None
        self._last_alpha_B: Optional[torch.Tensor] = None

        # I/O
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        if self.config.use_parallel_embedding:
            self.embed_tokens = ParallelEmbeddingV3(self.config.vocab_size, self.config.hidden_size, cast_to=self.forward_dtype)
        else:
            self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        # Output head projects to output vocab size (small classification set, e.g., digits)
        self.lm_head      = ColumnParallelLinearV3(self.config.hidden_size, self.config.output_vocab_size, bias=False)
        # Auxiliary reconstruction head over input BPE vocab (optional)
        if self.config.aux_recon_loss_weight > 0.0:
            self.recon_head = ColumnParallelLinearV3(self.config.hidden_size, self.config.vocab_size, bias=False)
        else:
            self.recon_head = None
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
            if self.config.original_seq_len is not None and (rope_max > self.config.original_seq_len) and (self.config.rope_factor > 1.0):
                self.rotary_emb = YarnRotaryEmbedding(
                    dim=self.config.qk_rope_head_dim,
                    max_position_embeddings=rope_max,
                    base=self.config.rope_theta,
                    original_seq_len=self.config.original_seq_len,
                    rope_factor=self.config.rope_factor,
                    beta_fast=self.config.beta_fast,
                    beta_slow=self.config.beta_slow,
                    mscale=self.config.mscale,
                )
            else:
                self.rotary_emb = RotaryEmbedding(dim=self.config.qk_rope_head_dim,
                                                  max_position_embeddings=rope_max,
                                                  base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            # For v3, we need extra positions for chitta slots, ego token, and plan tokens
            max_extra = self.config.num_chitta_slots + (1 if self.config.ego_token else 0) + self.config.num_plan_tokens
            total_positions = self.config.seq_len + self.puzzle_emb_len + max_extra
            self.embed_pos = CastedEmbedding(total_positions, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        # Reasoning layers per level
        self.C_level = HierarchicalReasoningModel4L_ACTV3ReasoningModule(layers=[HierarchicalReasoningModel4L_ACTV3Block(self.config) for _i in range(self.config.C_layers)])
        self.M_level = HierarchicalReasoningModel4L_ACTV3ReasoningModule(layers=[HierarchicalReasoningModel4L_ACTV3Block(self.config) for _i in range(self.config.M_layers)])
        self.B_level = HierarchicalReasoningModel4L_ACTV3ReasoningModule(layers=[HierarchicalReasoningModel4L_ACTV3Block(self.config) for _i in range(self.config.B_layers)])
        self.A_level = HierarchicalReasoningModel4L_ACTV3ReasoningModule(layers=[HierarchicalReasoningModel4L_ACTV3Block(self.config) for _i in range(self.config.A_layers)])

        # Optional compilation of steady modules
        if self.config.compile_modules:
            try:
                self.M_level = torch.compile(self.M_level)  # type: ignore[attr-defined]
                self.B_level = torch.compile(self.B_level)  # type: ignore[attr-defined]
                self.A_level = torch.compile(self.A_level)  # type: ignore[attr-defined]
                self.C_level = torch.compile(self.C_level)  # type: ignore[attr-defined]
            except Exception:
                pass

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
        return HierarchicalReasoningModel4L_ACTV3InnerCarry(
            z_C=torch.empty(batch_size, len_C, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_M=torch.empty(batch_size, len_M, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_B=torch.empty(batch_size, len_B, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_A=torch.empty(batch_size, len_A, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: 'HierarchicalReasoningModel4L_ACTV3InnerCarry'):
        return HierarchicalReasoningModel4L_ACTV3InnerCarry(
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

    def _seq_info_for(self, length: int):
        if hasattr(self, "rotary_emb"):
            cos, sin = self.rotary_emb()
            return dict(cos_sin=(cos[:length], sin[:length]))
        return dict(cos_sin=None)

    def _mtp_logits_from_base(self, base_logits: torch.Tensor) -> List[torch.Tensor]:
        # base_logits is the same tensor used for primary LM loss (already excludes prefixes)
        mtp_logits: List[torch.Tensor] = []
        if self.config.mtp_num_future <= 0:
            return mtp_logits
        L = base_logits.shape[-2]
        for k in range(1, self.config.mtp_num_future + 1):
            if L <= k:
                mtp_logits.append(base_logits[:, :0])
            else:
                # Predict t+k using features at t => compare logits at positions [0..L-k-1] to labels shifted by k
                mtp_logits.append(base_logits[:, :-k])
        return mtp_logits

    def _mtp_weights(self) -> List[float]:
        return [self.config.mtp_gamma ** (k - 1) for k in range(1, self.config.mtp_num_future + 1)]

    def forward(self, carry: HierarchicalReasoningModel4L_ACTV3InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel4L_ACTV3InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor], Optional[torch.Tensor]]:
        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"]).to(self.forward_dtype)

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
                        z_M = self.M_level(z_M, inj_M, **self._seq_info_for(z_M.shape[-2]))

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
                        hs = layer(hidden_states=hs, **self._seq_info_for(hs.shape[-2]))
                    z_B = hs[:, :z_B.shape[-2]]

                # A updates from B using A-shaped injection with [EGO] pooling
                base_len = input_embeddings.shape[-2]
                inj_A = torch.zeros_like(z_A)
                if self.config.ego_token:
                    # [EGO] token receives pooled B
                    inj_A[:, 0] = self._mean_pool(z_B)
                    # Remaining A tokens align to last base_len tokens of B (exclude plan tokens)
                    inj_A[:, 1:] = z_B[:, -base_len:]
                else:
                    # No ego token: align all A tokens to last base_len tokens of B
                    inj_A[:, :] = z_B[:, -z_A.shape[-2]:]
                a_len = z_A.shape[-2]
                a_mask = None
                if self.config.causal_in_a:
                    a_mask = torch.full((a_len, a_len), float("-inf"), device=z_A.device, dtype=z_A.dtype).triu_(1)
                z_A = self.A_level(z_A, inj_A, attn_mask=a_mask, **self._seq_info_for(z_A.shape[-2]))

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
        z_M = self.M_level(z_M, inj_M, **self._seq_info_for(z_M.shape[-2]))

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
            hs = layer(hidden_states=hs, **self._seq_info_for(hs.shape[-2]))
        z_B = hs[:, :z_B.shape[-2]]

        # A grad update from B with A-shaped injection
        base_len = input_embeddings.shape[-2]
        inj_A = torch.zeros_like(z_A)
        if self.config.ego_token:
            inj_A[:, 0] = self._mean_pool(z_B)
            inj_A[:, 1:] = z_B[:, -base_len:]
        else:
            inj_A[:, :] = z_B[:, -z_A.shape[-2]:]
        a_len = z_A.shape[-2]
        a_mask = None
        if self.config.causal_in_a:
            a_mask = torch.full((a_len, a_len), float("-inf"), device=z_A.device, dtype=z_A.dtype).triu_(1)
        z_A = self.A_level(z_A, inj_A, attn_mask=a_mask, **self._seq_info_for(z_A.shape[-2]))

        # Optional C write on grad step as well
        if (not self.config.freeze_c_writes) and (max(1, self.config.C_every_A) == 1):
            z_C = self._c_write_update(z_C, z_A, z_B, z_M, g_C_write)

        # Expose last g_C_write for regularization in outer wrapper
        self._last_g_C_write = g_C_write
        self._last_alpha_M = alpha_M
        self._last_alpha_B = alpha_B

        # LM/Q heads from A
        new_carry = HierarchicalReasoningModel4L_ACTV3InnerCarry(z_C=z_C.detach(), z_M=z_M.detach(), z_B=z_B.detach(), z_A=z_A.detach())
        # Exclude puzzle prefix and [EGO] token from LM outputs
        ego_offset = 1 if self.config.ego_token else 0
        logits = self.lm_head(z_A)[:, self.puzzle_emb_len + ego_offset:]

        # Q-head reads from [EGO] token (index 0 if enabled, else 0 anyway)
        q_index = 0
        q_logits = self.q_head(z_A[:, q_index]).to(torch.float32)

        # Multi-Token Prediction logits
        mtp_logits = self._mtp_logits_from_base(logits)
        # Auxiliary reconstruction logits
        recon_logits: Optional[torch.Tensor] = None
        if self.recon_head is not None:
            base_len = input_embeddings.shape[-2]
            recon_logits = self.recon_head(z_M[:, :base_len])
        
        return new_carry, logits, (q_logits[..., 0], q_logits[..., 1]), mtp_logits, recon_logits


class HierarchicalReasoningModel4L_ACTV3(nn.Module):
    """ACT wrapper for 4-level HRM v3 (DeepSeek-inspired improvements)."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel4L_ACTV3Config(**config_dict)
        self.inner = HierarchicalReasoningModel4L_ACTV3_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return HierarchicalReasoningModel4L_ACTV3Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, will be reset in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: HierarchicalReasoningModel4L_ACTV3Carry, batch: Dict[str, torch.Tensor]):
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model (use inference_mode when not training)
        if not self.training:
            with torch.inference_mode():
                new_inner_carry, logits, (q_halt_logits, q_continue_logits), mtp_logits, recon_logits = self.inner(new_inner_carry, new_current_data)
        else:
            new_inner_carry, logits, (q_halt_logits, q_continue_logits), mtp_logits, recon_logits = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        if recon_logits is not None:
            outputs["recon_logits"] = recon_logits

        if self.config.mtp_num_future > 0:
            outputs["mtp_logits"] = mtp_logits
            outputs["mtp_weights"] = self.inner._mtp_weights()
        
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

                # Compute target Q (bootstrapping) – explicitly grab q logits tuple
                _nc, _lg, (next_q_halt_logits, next_q_continue_logits), _ml, _rl = self.inner(new_inner_carry, new_current_data)
                
                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        # Surface last g_C_write for optional regularization in training loop
        if getattr(self.inner, "_last_g_C_write", None) is not None:
            outputs["g_C_write"] = self.inner._last_g_C_write
            
        # Surface FiLM alphas for debugging if available (as tensors to avoid graph breaks)
        if getattr(self.inner, "_last_alpha_M", None) is not None:
            outputs["alpha_M_mean"] = self.inner._last_alpha_M.mean().detach()
            outputs["alpha_B_mean"] = self.inner._last_alpha_B.mean().detach()

        return HierarchicalReasoningModel4L_ACTV3Carry(new_inner_carry, new_steps, halted, new_current_data), outputs


