from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class QuatfitConfig:
    # Basic parameters
    vocab_size: int = 256000
    hidden_size: int = 4096
    num_hidden_layers: int = 48
    num_dense_layers: int = 4  # The first N layers are dense
    num_attention_heads: int = 32
    num_key_value_groups: int = 4  # For Grouped-Query Attention (GQA)
    head_dim: int = 128
    
    # Layer specific parameters
    activation_fn: str = "swiglu"
    norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    
    # Positional encoding
    rope_theta: float = 10000.0
    use_yarn: bool = True
    max_position_embeddings: int = 32768  # Base context
    extended_max_position_embeddings: int = 1048576  # 1M context
    
    # Sparse MoE parameters
    is_moe: bool = True
    num_experts: int = 128
    num_shared_experts: int = 1
    top_k: int = 4
    expert_ffn_dim: int = 6144
    dense_ffn_dim: Optional[int] = None
    aux_loss_free_balancing: bool = True
    
    # Attention details
    attention_type: str = "gqa"  # "gqa" or "mla"
    mla_kv_compression_dim: int = 512
    mla_query_compression_dim: int = 1536
    mla_rope_head_dim: int = 64
    
    # Sliding window and landmarks
    sliding_window_size: int = 4096
    landmark_interval: int = 512
    
    # Adaptive Computation
    use_adaptive_computation: bool = True
    adaptive_exit_checkpoints: List[int] = field(default_factory=lambda: [12, 24, 36]) # Checkpoints at specific layers
    layer_dropout_prob: float = 0.1
    
    # Chain-of-Thought Verifier
    use_cot_verifier: bool = False
    verifier_hidden_size: int = 512
    verifier_num_layers: int = 4
    
    # Hierarchical Memory (Google Titans inspired)
    use_hierarchical_memory: bool = True
    memory_tiers_active: List[int] = field(default_factory=lambda: [1, 2])  # Tiers: 1 = Working, 2 = Persistent, 3 = Archive
    surprise_threshold_persistent: float = 0.5
    surprise_threshold_archive: float = 0.2
    max_archive_size: int = 100000
    
    # Multi-Token Prediction (MTP)
    use_mtp: bool = True
    mtp_depth: int = 4  # Predict next K tokens
    mtp_loss_weight: float = 0.3

    # Embedding factoring
    use_factored_embeddings: bool = True
    factored_embed_dim: int = 512

    # Precision
    precision: str = "fp8"

    @classmethod
    def get_preset_config(cls, size: str) -> "QuatfitConfig":
        size = size.lower()
        if size == "nano":
            return cls(
                hidden_size=2048,
                num_hidden_layers=24,
                num_dense_layers=24,  # fully dense
                num_attention_heads=16,
                num_key_value_groups=4,
                head_dim=128,
                is_moe=False,
                use_adaptive_computation=False,
                use_hierarchical_memory=False,
                use_cot_verifier=False,
                use_mtp=True,
                mtp_depth=2,
                precision="bf16",
                use_factored_embeddings=True,
                factored_embed_dim=256
            )
        elif size == "mini":
            return cls(
                hidden_size=3072,
                num_hidden_layers=32,
                num_dense_layers=4,
                num_attention_heads=32,
                num_key_value_groups=4,
                head_dim=96,
                is_moe=True,
                num_experts=64,
                top_k=4,
                expert_ffn_dim=4096,
                use_adaptive_computation=True,
                adaptive_exit_checkpoints=[8, 16, 24],
                use_hierarchical_memory=True,
                memory_tiers_active=[1],  # Working memory only
                use_cot_verifier=False,
                use_mtp=True,
                mtp_depth=3,
                precision="fp8"
            )
        elif size == "base":
            return cls(
                hidden_size=4096,
                num_hidden_layers=48,
                num_dense_layers=4,
                num_attention_heads=32,
                num_key_value_groups=4,
                head_dim=128,
                is_moe=True,
                num_experts=128,
                top_k=4,
                expert_ffn_dim=6144,
                use_adaptive_computation=True,
                adaptive_exit_checkpoints=[12, 24, 36],
                use_hierarchical_memory=True,
                memory_tiers_active=[1, 2],  # Working + Persistent
                use_cot_verifier=False,
                use_mtp=True,
                mtp_depth=4,
                precision="fp8"
            )
        elif size == "pro":
            return cls(
                hidden_size=6144,
                num_hidden_layers=64,
                num_dense_layers=4,
                num_attention_heads=64,
                num_key_value_groups=4,
                head_dim=96,
                is_moe=True,
                num_experts=256,
                top_k=8,
                expert_ffn_dim=8192,
                use_adaptive_computation=True,
                adaptive_exit_checkpoints=[16, 32, 48],
                use_hierarchical_memory=True,
                memory_tiers_active=[1, 2, 3],  # All 3 tiers
                use_cot_verifier=True,
                use_mtp=True,
                mtp_depth=4,
                precision="fp8"
            )
        elif size == "ultra":
            return cls(
                hidden_size=7168,
                num_hidden_layers=72,
                num_dense_layers=6,
                num_attention_heads=128,
                num_key_value_groups=8,
                head_dim=56,
                is_moe=True,
                num_experts=256,
                top_k=8,
                expert_ffn_dim=12288,
                use_adaptive_computation=True,
                adaptive_exit_checkpoints=[18, 36, 54],
                use_hierarchical_memory=True,
                memory_tiers_active=[1, 2, 3],  # All 3 tiers
                use_cot_verifier=True,
                use_mtp=True,
                mtp_depth=4,
                precision="fp8"
            )
        else:
            raise ValueError(f"Unknown preset size: {size}. Options: 'nano', 'mini', 'base', 'pro', 'ultra'")
