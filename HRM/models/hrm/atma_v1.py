 """
ATMA v1 - Kaggle-optimized version of HRM ACT V2
Modified for Kaggle compatibility:
- Fixed linter issues with tensor buffer access
- Commented out debug film_stats logging to avoid tensor indexing issues
- Optimized for Kaggle's GPU memory constraints
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import logging

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.hrm.memory import ChittaMemory, ChittaMemoryBank
from models.hrm.vedic_modules import AhamkaraModule, ChittaModule, BuddhiModule, ManasModule

logger = logging.getLogger(__name__)


@dataclass
class ATMA_V1InnerCarry:
    """Internal carry state for the four Vedic modules.
    
    Vedic Architecture:
    - a_state: Ahamkara (अहंकार) - ego-function/identity state
    - c_state: Chitta (चित्त) - consciousness repository state  
    - b_state: Buddhi (बुद्धि) - discriminative intelligence state
    - m_state: Manas (मनस्) - sensory-motor mind state
    """
    a_state: torch.Tensor  # Ahamkara state
    c_state: torch.Tensor  # Chitta state
    b_state: torch.Tensor  # Buddhi state
    m_state: torch.Tensor  # Manas state
    chitta_memory: ChittaMemory  # Persistent memory


@dataclass
class ATMA_V1Carry:
    """Complete carry state including ACT control variables."""
    inner_carry: ATMA_V1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class ATMA_V1Config(BaseModel):
    """Configuration for four-layer Vedic HRM architecture."""
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int
    
    # Module cycles (timescales)
    A_cycles: int = 1  # Ahamkara - slowest
    C_cycles: int = 1  # Chitta  
    B_cycles: int = 2  # Buddhi
    M_cycles: int = 2  # Manas - fastest
    
    # Module layers
    A_layers: int = 2  # Ahamkara layers
    C_layers: int = 2  # Chitta layers
    B_layers: int = 4  # Buddhi layers
    M_layers: int = 4  # Manas layers
    
    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    
    # Chitta memory config
    chitta_memory_slots: int = 32
    chitta_memory_dim: int = 512
    
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float
    
    # Ablation flags
    disable_ahamkara: bool = False
    disable_chitta: bool = False
    
    # Debug mode
    debug_mode: bool = False
    
    # Memory parameters
    memory_decay_rate: float = 0.0
    memory_write_threshold: float = 0.5
    
    forward_dtype: str = "bfloat16"





class ATMA_V1Block(nn.Module):
    """Transformer block used by all Vedic modules."""
    
    def __init__(self, config: ATMA_V1Config) -> None:
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
        # Post Norm architecture
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), 
            variance_epsilon=self.norm_eps
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states), 
            variance_epsilon=self.norm_eps
        )
        return hidden_states


class ATMA_V1_Inner(nn.Module):
    """Inner model implementing the four-layer Vedic architecture."""
    
    def __init__(self, config: ATMA_V1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        
        # Debug logging
        self.debug_logs = {}
        
        # I/O layers
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        
        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, 
            self.config.hidden_size, 
            init_std=embed_init_std, 
            cast_to=self.forward_dtype
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        # Q-heads for hierarchical halting
        self.q_halt_m = CastedLinear(self.config.hidden_size, 2, bias=True)  # Manas halt
        self.q_halt_b = CastedLinear(self.config.hidden_size, 2, bias=True)  # Buddhi halt
        self.q_commit_c = CastedLinear(self.config.hidden_size, 2, bias=True)  # Chitta commit
        self.q_update_a = CastedLinear(self.config.hidden_size, 2, bias=True)  # Ahamkara update
        
        # Puzzle embeddings
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
        if self.config.puzzle_emb_ndim > 0:
            # 🔥 SMOKING GUN TEST: Use trainable CastedEmbedding instead of CastedSparseEmbedding
            # CastedSparseEmbedding uses nn.Buffer (non-trainable), we need nn.Parameter (trainable)
            self.puzzle_emb = CastedEmbedding(
                self.config.num_puzzle_identifiers, 
                self.config.puzzle_emb_ndim,
                init_std=0, 
                cast_to=self.forward_dtype
            )
            
        # Position encodings
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len, 
                self.config.hidden_size, 
                init_std=embed_init_std, 
                cast_to=self.forward_dtype
            )
            
        # Initialize the four Vedic modules
        self.ahamkara = AhamkaraModule(self.config) if not self.config.disable_ahamkara else None
        self.chitta = ChittaModule(self.config) if not self.config.disable_chitta else None
        self.buddhi = BuddhiModule(self.config)
        self.manas = ManasModule(self.config)
        
        # Initial states
        self.a_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.c_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.b_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.m_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        
        # Initialize Q-heads
        with torch.no_grad():
            for q_head in [self.q_halt_m, self.q_halt_b, self.q_commit_c, self.q_update_a]:
                q_head.weight.zero_()
                if q_head.bias is not None:
                    q_head.bias.fill_(-5)
                    
        # Initialize memory bank for advanced operations
        self.memory_bank = ChittaMemoryBank(
            num_slots=config.chitta_memory_slots,
            key_dim=config.chitta_memory_dim,
            value_dim=config.chitta_memory_dim
        )
        
        # Log parameter count
        self._log_parameter_count()
                
    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        """Generate input embeddings with puzzle and position encodings."""
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))
        
        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            # 🔥 SMOKING GUN TEST: Handle regular CastedEmbedding (not sparse)
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)  # Shape: [batch, puzzle_emb_ndim]
            
            # Reshape to match expected dimensions
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
                
            # Ensure both tensors are on the same device before concatenating
            puzzle_emb_reshaped = puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size)
            if puzzle_emb_reshaped.device != embedding.device:
                puzzle_emb_reshaped = puzzle_emb_reshaped.to(embedding.device)
            embedding = torch.cat(
                (puzzle_emb_reshaped, embedding), 
                dim=-2
            )
            
        # Position embeddings
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
            
        return self.embed_scale * embedding
        
    def _log_parameter_count(self):
        """Log parameter counts for each module."""
        total_params = 0
        param_counts = {}
        
        # Count parameters by module
        modules = [
            ('embedding', [self.embed_tokens, self.embed_pos if hasattr(self, 'embed_pos') else None]),
            ('ahamkara', [self.ahamkara]),
            ('chitta', [self.chitta, self.memory_bank]),
            ('buddhi', [self.buddhi]),
            ('manas', [self.manas]),
            ('heads', [self.lm_head, self.q_halt_m, self.q_halt_b, self.q_commit_c, self.q_update_a])
        ]
        
        for name, module_list in modules:
            count = 0
            for module in module_list:
                if module is not None:
                    count += sum(p.numel() for p in module.parameters())
            param_counts[name] = count
            total_params += count
            
        # Log results
        logger.info("=" * 50)
        logger.info("ATMA V1 Parameter Count:")
        logger.info("=" * 50)
        for name, count in param_counts.items():
            logger.info(f"{name:12s}: {count:,} ({count/1e6:.2f}M)")
        logger.info("-" * 50)
        logger.info(f"{'TOTAL':12s}: {total_params:,} ({total_params/1e6:.2f}M)")
        logger.info("=" * 50)
        
        # Assert under budget
        assert total_params < 100e6, f"Total parameters {total_params/1e6:.2f}M exceeds 100M limit"
        
    def _collect_debug_stats(self, a_state, c_state, b_state, m_state):
        """Collect debug statistics for each module."""
        with torch.no_grad():
            # Module activation statistics
            self.debug_logs['module_stats'] = {
                'ahamkara': {
                    'mean': a_state.mean().item(),
                    'std': a_state.std().item(),
                    'sparsity': (a_state.abs() < 0.01).float().mean().item()
                },
                'chitta': {
                    'mean': c_state.mean().item(),
                    'std': c_state.std().item(),
                    'sparsity': (c_state.abs() < 0.01).float().mean().item()
                },
                'buddhi': {
                    'mean': b_state.mean().item(),
                    'std': b_state.std().item(),
                    'sparsity': (b_state.abs() < 0.01).float().mean().item()
                },
                'manas': {
                    'mean': m_state.mean().item(),
                    'std': m_state.std().item(),
                    'sparsity': (m_state.abs() < 0.01).float().mean().item()
                }
            }
            
            # FiLM parameter tracking (if Ahamkara enabled)
            # Note: Commented out for Kaggle compatibility - linter issues with tensor access
            # if self.ahamkara is not None and hasattr(self.ahamkara, 'film_stats'):
            #     # Get FiLM statistics as a list
            #     film_stats_list = self.ahamkara.film_stats.tolist()
            #     self.debug_logs['film_stats'] = {
            #         'chitta_gamma': film_stats_list[0],
            #         'chitta_beta': film_stats_list[1],
            #         'buddhi_gamma': film_stats_list[2],
            #         'buddhi_beta': film_stats_list[3],
            #         'manas_gamma': film_stats_list[4],
            #         'manas_beta': film_stats_list[5],
            #     }
        
    def empty_carry(self, batch_size: int):
        """Create empty carry state."""
        device = self.a_init.device
        seq_len = self.config.seq_len + self.puzzle_emb_len
        
        # Initialize Chitta memory using memory bank
        chitta_memory = self.memory_bank.init_memory(
            batch_size=batch_size,
            device=device,
            dtype=self.forward_dtype
        )
        
        return ATMA_V1InnerCarry(
            a_state=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            c_state=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            b_state=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            m_state=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            chitta_memory=chitta_memory
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: ATMA_V1InnerCarry):
        """Reset carry state for new sequences."""
        # Ensure init tensors are on the same device as the carry states
        device = carry.a_state.device
        batch_size = reset_flag.shape[0]
        seq_len = carry.a_state.shape[1]
        
        # Move reset_flag to the same device as carry states
        reset_flag = reset_flag.to(device)
        
        # Expand init tensors to match carry state shapes
        a_init_expanded = self.a_init.to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        c_init_expanded = self.c_init.to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        b_init_expanded = self.b_init.to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        m_init_expanded = self.m_init.to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        return ATMA_V1InnerCarry(
            a_state=torch.where(reset_flag.view(-1, 1, 1), a_init_expanded, carry.a_state),
            c_state=torch.where(reset_flag.view(-1, 1, 1), c_init_expanded, carry.c_state),
            b_state=torch.where(reset_flag.view(-1, 1, 1), b_init_expanded, carry.b_state),
            m_state=torch.where(reset_flag.view(-1, 1, 1), m_init_expanded, carry.m_state),
            chitta_memory=carry.chitta_memory  # Memory persists across sequences
        )
        
    def forward(self, 
                carry: ATMA_V1InnerCarry, 
                batch: Dict[str, torch.Tensor]):
        """Forward pass implementing four-layer Vedic architecture.
        
        Information Flow:
        1. Ahamkara (slowest) → FiLM modulation → all modules
        2. Chitta → context/memory → Buddhi, Manas
        3. Buddhi → control → Manas
        4. Manas (fastest) → sensory processing → Buddhi
        """
        
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        
        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        
        # Extract states
        a_state, c_state, b_state, m_state = carry.a_state, carry.c_state, carry.b_state, carry.m_state
        chitta_memory = carry.chitta_memory
        
        # TRUE HIERARCHICAL NESTING (HRM-STYLE): Cross-frequency coupling
        # Most computation NO_GRAD, only final refinement has gradients
        with torch.no_grad():
            # TRUE NESTING: Ahamkara (1x) → Chitta (2x) → Buddhi (4x) → Manas (8x)
            for a_step in range(self.config.A_cycles):  # Slowest: 1x
                for c_step in range(self.config.C_cycles):  # 2x per Ahamkara
                    for b_step in range(self.config.B_cycles):  # 4x per Chitta
                        for m_step in range(self.config.M_cycles):  # 8x per Buddhi (fastest)
                            
                            # Skip final step (needs gradients)
                            if not ((a_step == self.config.A_cycles - 1) and 
                                   (c_step == self.config.C_cycles - 1) and
                                   (b_step == self.config.B_cycles - 1) and 
                                   (m_step == self.config.M_cycles - 1)):
                                
                                # Manas: Fast sensory-motor processing
                                buddhi_control = self.buddhi.get_manas_control(b_state) if self.buddhi is not None else torch.zeros_like(m_state)
                                chitta_context, _ = self.chitta.read_memory(m_state, chitta_memory) if self.chitta is not None else (torch.zeros_like(m_state), None)
                                film_manas = self.ahamkara.get_film_params(a_state)["manas"] if self.ahamkara is not None else None
                                
                                m_state = self.manas(
                                    m_state,
                                    input_embeddings + a_state + c_state + b_state,  # HRM-style feedback
                                    film_manas,
                                    buddhi_control=buddhi_control + chitta_context,
                                    **seq_info
                                )
                        
                        # Buddhi: After all Manas cycles for this step
                        if not ((a_step == self.config.A_cycles - 1) and 
                               (c_step == self.config.C_cycles - 1) and
                               (b_step == self.config.B_cycles - 1)):
                            
                            chitta_context, _ = self.chitta.read_memory(b_state, chitta_memory) if self.chitta is not None else (torch.zeros_like(b_state), None)
                            film_buddhi = self.ahamkara.get_film_params(a_state)["buddhi"] if self.ahamkara is not None else None
                            
                            b_state = self.buddhi(
                                b_state, 
                                input_embeddings + a_state + c_state + m_state,  # Include Manas feedback
                                film_buddhi,
                                chitta_context=chitta_context,
                                **seq_info
                            )
                    
                    # Chitta: After all Buddhi cycles for this step
                    if not ((a_step == self.config.A_cycles - 1) and 
                           (c_step == self.config.C_cycles - 1)) and self.chitta is not None:
                        
                        c_context, attn_weights = self.chitta.read_memory(
                            c_state, chitta_memory, 
                            return_attention=self.config.debug_mode
                        )
                        film_chitta = self.ahamkara.get_film_params(a_state)["chitta"] if self.ahamkara is not None else None
                        
                        c_state = self.chitta(c_state, c_context + b_state + m_state, film_chitta, **seq_info)  # Include lower-level feedback
                        new_memory = self.chitta.write_memory(c_state, chitta_memory)
                        if isinstance(new_memory, tuple):
                            chitta_memory, _ = new_memory
                        else:
                            chitta_memory = new_memory
                
                # Ahamkara: After all Chitta cycles for this step
                if not (a_step == self.config.A_cycles - 1) and self.ahamkara is not None:
                    a_state = self.ahamkara(a_state, input_embeddings + c_state + b_state + m_state, **seq_info)  # All lower-level feedback

        # FINAL REFINEMENT WITH GRADIENTS (like HRM's single gradient step)
        # Final nested refinement - single pass with gradients
        
        # Final Manas step WITH gradients (innermost/fastest)
        buddhi_control = self.buddhi.get_manas_control(b_state) if self.buddhi is not None else torch.zeros_like(m_state)
        chitta_context, _ = self.chitta.read_memory(m_state, chitta_memory) if self.chitta is not None else (torch.zeros_like(m_state), None)
        film_manas = self.ahamkara.get_film_params(a_state)["manas"] if self.ahamkara is not None else None
        
        m_state = self.manas(
            m_state,
            input_embeddings + a_state + c_state + b_state,  # All higher-level feedback
            film_manas,
            buddhi_control=buddhi_control + chitta_context,
            **seq_info
        )
        
        # Final Buddhi step WITH gradients
        chitta_context, _ = self.chitta.read_memory(b_state, chitta_memory) if self.chitta is not None else (torch.zeros_like(b_state), None)
        film_buddhi = self.ahamkara.get_film_params(a_state)["buddhi"] if self.ahamkara is not None else None
        
        b_state = self.buddhi(
            b_state, 
            input_embeddings + a_state + c_state + m_state,  # Include Manas feedback
            film_buddhi,
            chitta_context=chitta_context,
            **seq_info
        )
        
        # Final Chitta step WITH gradients
        if self.chitta is not None:
            c_context, attn_weights = self.chitta.read_memory(
                c_state, chitta_memory, 
                return_attention=self.config.debug_mode
            )
            film_chitta = self.ahamkara.get_film_params(a_state)["chitta"] if self.ahamkara is not None else None
            
            c_state = self.chitta(c_state, c_context + b_state + m_state, film_chitta, **seq_info)
            new_memory = self.chitta.write_memory(c_state, chitta_memory)
            if isinstance(new_memory, tuple):
                chitta_memory, _ = new_memory
            else:
                chitta_memory = new_memory
            
            if self.config.debug_mode and attn_weights is not None:
                self.debug_logs['chitta_attention'] = attn_weights
        
        # Final Ahamkara step WITH gradients (outermost/slowest)
        if self.ahamkara is not None:
            a_state = self.ahamkara(a_state, input_embeddings + c_state + b_state + m_state, **seq_info)
            
        # Collect module statistics if debug mode
        if self.config.debug_mode:
            self._collect_debug_stats(a_state, c_state, b_state, m_state)
            
        # Generate outputs from Buddhi (main reasoning module)
        output = self.lm_head(b_state)[:, self.puzzle_emb_len:]
        
        # Compute Q-heads for hierarchical halting
        q_heads = {
            "q_halt_m": self.q_halt_m(m_state[:, 0]).to(torch.float32),      # Manas halt
            "q_halt_b": self.q_halt_b(b_state[:, 0]).to(torch.float32),      # Buddhi halt
            "q_commit_c": self.q_commit_c(c_state[:, 0]).to(torch.float32),  # Chitta commit
            "q_update_a": self.q_update_a(a_state[:, 0]).to(torch.float32),  # Ahamkara update
        }
        
        # Update Chitta memory based on Q-head decision
        write_stats = None
        if self.chitta is not None and self.training:
            # Check if we should commit to memory
            commit_prob = torch.sigmoid(q_heads["q_commit_c"][..., 0])
            should_write = (commit_prob > self.config.memory_write_threshold).any()
            
            if should_write:
                result = self.chitta.write_memory(
                    c_state, 
                    chitta_memory,
                    threshold=self.config.memory_write_threshold
                )
                if isinstance(result, tuple):
                    chitta_memory, write_stats = result
                else:
                    chitta_memory = result
                
                # Apply memory consolidation periodically
                if hasattr(self, '_step_count'):
                    self._step_count += 1
                    chitta_memory = self.memory_bank.consolidate_memories(
                        chitta_memory, self._step_count
                    )
                else:
                    self._step_count = 0
                    
        if self.config.debug_mode and write_stats is not None:
            self.debug_logs['memory_write_stats'] = write_stats
                
        # Create new carry
        new_carry = ATMA_V1InnerCarry(
            a_state=a_state.detach(),
            c_state=c_state.detach(),
            b_state=b_state.detach(),
            m_state=m_state.detach(),
            chitta_memory=chitta_memory
        )
        
        # Return debug logs if requested
        if self.config.debug_mode:
            return new_carry, output, q_heads, self.debug_logs
        return new_carry, output, q_heads
        

class ATMA_V1(nn.Module):
    """Four-layer Vedic HRM with ACT wrapper.
    
    Implements the classical Vedic four-layer model of consciousness:
    - Ahamkara (अहंकार): Identity/ego-function providing slow modulation
    - Chitta (चित्त): Consciousness repository with persistent memory
    - Buddhi (बुद्धि): Discriminative intelligence for reasoning
    - Manas (मनस्): Sensory-motor mind for rapid processing
    
    This architecture extends HRM's dual-module approach to capture the full
    sophistication of Vedic consciousness theory while maintaining computational
    efficiency and training stability.
    """
    
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = ATMA_V1Config(**config_dict)
        self.inner = ATMA_V1_Inner(self.config)
        
    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb if hasattr(self.inner, 'puzzle_emb') else None
        
    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        """Initialize carry state for new batch."""
        batch_size = batch["inputs"].shape[0]
        
        return ATMA_V1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=batch["inputs"].device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=batch["inputs"].device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, 
                carry: ATMA_V1Carry, 
                batch: Dict[str, torch.Tensor]) -> Tuple[ATMA_V1Carry, Dict[str, torch.Tensor]]:
        """Forward pass with hierarchical ACT halting."""
        
        # Update data and carry for non-halted sequences
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) 
            for k, v in carry.current_data.items()
        }
        
        # Forward inner model
        inner_result = self.inner(new_inner_carry, new_current_data)
        if len(inner_result) == 4:  # Debug mode
            new_inner_carry, logits, q_heads, debug_logs = inner_result
        else:
            new_inner_carry, logits, q_heads = inner_result
            debug_logs = None
        
        # Extract individual Q-values
        q_halt_m_logits, q_continue_m_logits = q_heads["q_halt_m"][..., 0], q_heads["q_halt_m"][..., 1]
        q_halt_b_logits, q_continue_b_logits = q_heads["q_halt_b"][..., 0], q_heads["q_halt_b"][..., 1]
        q_commit_c_logits, q_no_commit_c_logits = q_heads["q_commit_c"][..., 0], q_heads["q_commit_c"][..., 1]
        q_update_a_logits, q_no_update_a_logits = q_heads["q_update_a"][..., 0], q_heads["q_update_a"][..., 1]
        
        outputs = {
            "logits": logits,
            "q_halt_b_logits": q_halt_b_logits,  # Main halting signal
            "q_continue_b_logits": q_continue_b_logits,
            # Additional Q-heads for logging/analysis
            "q_halt_m_logits": q_halt_m_logits,
            "q_commit_c_logits": q_commit_c_logits,
            "q_update_a_logits": q_update_a_logits,
        }
        
        with torch.no_grad():
            # Step counter
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            # Hierarchical halting logic
            halted = is_last_step
            
            if self.training and (self.config.halt_max_steps > 1):
                # Buddhi-level halting (main control)
                halted = halted | (q_halt_b_logits > q_continue_b_logits)
                
                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_b_logits) < self.config.halt_exploration_prob) * \
                                torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)
                
                # Compute target Q for bootstrapping
                next_result = self.inner(new_inner_carry, new_current_data)
                # Handle debug mode which returns 4 items
                if len(next_result) == 4:
                    _, _, next_q_heads, _ = next_result
                else:
                    _, _, next_q_heads = next_result
                next_q_halt_b, next_q_continue_b = next_q_heads["q_halt_b"][..., 0], next_q_heads["q_halt_b"][..., 1]
                
                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(is_last_step, next_q_halt_b, torch.maximum(next_q_halt_b, next_q_continue_b))
                )
                
        # Include debug logs if available
        if debug_logs is not None:
            outputs["debug_logs"] = debug_logs
                
        return ATMA_V1Carry(
            new_inner_carry, new_steps, halted, new_current_data
        ), outputs