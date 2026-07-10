import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from quatfit.model.config import QuatfitConfig
from quatfit.serving.paged_cache import PagedKVCacheManager
from quatfit.model.dynamic_precision import DynamicLinear

class YaRNScaledRotaryEmbedding(nn.Module):
    """
    YaRN (Yet another RoPE extensioN) scaled Rotary Position Embeddings.
    Allows extension of the context length dynamically with minimal degradation.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        
        # Calculate inv_freq
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for sin/cos
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        # Scale positions/frequencies with YaRN scaling
        # In YaRN, we scale the positions or base frequencies by 1/scale
        t = t / self.scale
        
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        
        return (
            self.cos_cached[:seq_len].to(x.device),
            self.sin_cached[:seq_len].to(x.device),
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # cos, sin: [seq_len, dim] -> [1, 1, seq_len, dim] or similar
    cos = cos[position_ids].unsqueeze(1) # [batch, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1) # [batch, 1, seq_len, dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seqlen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seqlen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seqlen, head_dim)


class QuatfitAttention(nn.Module):
    """
    Unified Attention module supporting Grouped-Query Attention (GQA)
    and Multi-head Latent Attention (MLA), plus sliding window and global landmarks.
    """
    def __init__(self, config: QuatfitConfig, layer_idx: int = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.attention_type = config.attention_type
        
        # YaRN scaled RoPE
        # If config size context is extended, adjust scaling factor
        scale_factor = 1.0
        if config.extended_max_position_embeddings > config.max_position_embeddings:
            scale_factor = float(config.extended_max_position_embeddings) / config.max_position_embeddings
        
        # Decide RoPE dimension
        if self.attention_type == "mla":
            self.rope_dim = config.mla_rope_head_dim
        else:
            self.rope_dim = self.head_dim
            
        self.rotary_emb = YaRNScaledRotaryEmbedding(
            dim=self.rope_dim,
            max_position_embeddings=config.extended_max_position_embeddings,
            base=config.rope_theta,
            scale=scale_factor
        )

        if self.attention_type == "gqa":
            self.num_kv_heads = config.num_key_value_groups
            self.num_queries_per_kv = self.num_heads // self.num_kv_heads
            
            self.q_proj = DynamicLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False, precision=config.precision)
            self.k_proj = DynamicLinear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False, precision=config.precision)
            self.v_proj = DynamicLinear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False, precision=config.precision)
            self.o_proj = DynamicLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False, precision=config.precision)
            
        elif self.attention_type == "mla":
            # MLA compression parameters
            self.kv_compressed_dim = config.mla_kv_compression_dim
            self.query_compressed_dim = config.mla_query_compression_dim
            
            # Query compression
            self.q_down_proj = DynamicLinear(self.hidden_size, self.query_compressed_dim, bias=False, precision=config.precision)
            self.q_down_norm = nn.LayerNorm(self.query_compressed_dim)
            self.q_up_proj = DynamicLinear(self.query_compressed_dim, self.num_heads * self.head_dim, bias=False, precision=config.precision)
            
            # KV compression
            self.kv_down_proj = DynamicLinear(self.hidden_size, self.kv_compressed_dim, bias=False, precision=config.precision)
            self.kv_down_norm = nn.LayerNorm(self.kv_compressed_dim)
            
            # Key content & position projection
            self.k_up_proj = DynamicLinear(self.kv_compressed_dim, self.num_heads * self.head_dim, bias=False, precision=config.precision)
            self.v_up_proj = DynamicLinear(self.kv_compressed_dim, self.num_heads * self.head_dim, bias=False, precision=config.precision)
            
            # Decoupled RoPE projections
            self.q_rope_proj = DynamicLinear(self.query_compressed_dim, self.num_heads * self.rope_dim, bias=False, precision=config.precision)
            self.k_rope_proj = DynamicLinear(self.hidden_size, self.rope_dim, bias=False, precision=config.precision) # standard RoPE projection from input
            
            self.o_proj = DynamicLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False, precision=config.precision)

        self._cached_mask = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        if self.attention_type == "gqa":
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            
            # Reshape for multi-head attention
            q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            
            kv_seq_len = k.shape[-2]
            if past_key_value is not None:
                if isinstance(past_key_value, PagedKVCacheManager):
                    kv_seq_len += past_key_value.context_lengths[0]
                else:
                    kv_seq_len += past_key_value[0].shape[-2]
                
            cos, sin = self.rotary_emb(q, seq_len=kv_seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            
            if past_key_value is not None:
                if isinstance(past_key_value, PagedKVCacheManager):
                    cache = past_key_value
                    for b in range(bsz):
                        start_pos = cache.context_lengths[b]
                        seq_len_to_add = k.shape[2]
                        for i in range(seq_len_to_add):
                            pos = start_pos + i
                            block_idx = cache.block_tables[b][pos // cache.block_size]
                            offset = pos % cache.block_size
                            cache.key_cache[block_idx, :, offset, :] = k[b, :, i, :]
                            cache.value_cache[block_idx, :, offset, :] = v[b, :, i, :]
                        cache.context_lengths[b] += seq_len_to_add
                        
                    full_k, full_v = [], []
                    for b in range(bsz):
                        seq_len = cache.context_lengths[b]
                        blocks = cache.block_tables[b]
                        k_b, v_b = [], []
                        for i in range(seq_len):
                            block_idx = blocks[i // cache.block_size]
                            offset = i % cache.block_size
                            k_b.append(cache.key_cache[block_idx, :, offset, :])
                            v_b.append(cache.value_cache[block_idx, :, offset, :])
                        full_k.append(torch.stack(k_b, dim=1))
                        full_v.append(torch.stack(v_b, dim=1))
                    k = torch.stack(full_k, dim=0)
                    v = torch.stack(full_v, dim=0)
                    
                    new_past_key_value = cache if use_cache else None
                else:
                    k = torch.cat([past_key_value[0], k], dim=-2)
                    v = torch.cat([past_key_value[1], v], dim=-2)
                    new_past_key_value = (k, v) if use_cache else None
            else:
                new_past_key_value = (k, v) if use_cache else None
            
            # Repeat keys/values to match query head count
            k = repeat_kv(k, self.num_queries_per_kv)
            v = repeat_kv(v, self.num_queries_per_kv)
            
        elif self.attention_type == "mla":
            # Query path
            q_compressed = self.q_down_norm(self.q_down_proj(hidden_states))
            q_content = self.q_up_proj(q_compressed).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            q_rope = self.q_rope_proj(q_compressed).view(bsz, q_len, self.num_heads, self.rope_dim).transpose(1, 2)
            
            # Key-Value path
            kv_compressed = self.kv_down_norm(self.kv_down_proj(hidden_states)).unsqueeze(1)
            
            # Decoupled RoPE Key
            k_rope = self.k_rope_proj(hidden_states).view(bsz, q_len, 1, self.rope_dim).transpose(1, 2)
            
            # Set up positional encodings
            kv_seq_len = q_len
            if past_key_value is not None:
                if isinstance(past_key_value, PagedKVCacheManager):
                    kv_seq_len += past_key_value.context_lengths[0]
                else:
                    kv_seq_len += past_key_value[0].shape[-2] # past_key_value[0] holds kv_compressed
                
            cos, sin = self.rotary_emb(q_rope, seq_len=kv_seq_len)
            
            # Apply rotary to decoupled parts
            q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin, position_ids)
            
            # Cache compressed representations to save memory
            if past_key_value is not None:
                if isinstance(past_key_value, PagedKVCacheManager):
                    cache = past_key_value
                    for b in range(bsz):
                        start_pos = cache.context_lengths[b]
                        seq_len_to_add = kv_compressed.shape[2]
                        for i in range(seq_len_to_add):
                            pos = start_pos + i
                            block_idx = cache.block_tables[b][pos // cache.block_size]
                            offset = pos % cache.block_size
                            cache.key_cache[block_idx, :, offset, :kv_compressed.shape[-1]] = kv_compressed[b, :, i, :]
                            cache.value_cache[block_idx, :, offset, :k_rope.shape[-1]] = k_rope[b, :, i, :]
                        cache.context_lengths[b] += seq_len_to_add
                        
                    full_kv, full_k_rope = [], []
                    for b in range(bsz):
                        seq_len = cache.context_lengths[b]
                        blocks = cache.block_tables[b]
                        kv_b, k_rope_b = [], []
                        for i in range(seq_len):
                            block_idx = blocks[i // cache.block_size]
                            offset = i % cache.block_size
                            kv_b.append(cache.key_cache[block_idx, :, offset, :kv_compressed.shape[-1]])
                            k_rope_b.append(cache.value_cache[block_idx, :, offset, :k_rope.shape[-1]])
                        full_kv.append(torch.stack(kv_b, dim=1))
                        full_k_rope.append(torch.stack(k_rope_b, dim=1))
                    
                    cached_kv_compressed = torch.stack(full_kv, dim=0)
                    cached_k_rope = torch.stack(full_k_rope, dim=0)
                    new_past_key_value = cache if use_cache else None
                else:
                    # Concatenate along sequence dimension
                    cached_kv_compressed = torch.cat([past_key_value[0], kv_compressed], dim=-2)
                    cached_k_rope = torch.cat([past_key_value[1], k_rope], dim=-2)
                    new_past_key_value = (cached_kv_compressed, cached_k_rope) if use_cache else None
            else:
                cached_kv_compressed = kv_compressed
                cached_k_rope = k_rope
                new_past_key_value = (cached_kv_compressed, cached_k_rope) if use_cache else None
            
            # Reconstruct Full K & V content for attention
            # cached_kv_compressed: [bsz, 1, kv_seq_len, kv_compressed_dim]
            k_content = self.k_up_proj(cached_kv_compressed.squeeze(1)).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_up_proj(cached_kv_compressed.squeeze(1)).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Re-expand k_rope for all heads
            k_rope_expanded = cached_k_rope.expand(-1, self.num_heads, -1, -1)
            
            # Combine content and position for Q and K
            # Q is size [bsz, num_heads, q_len, head_dim + rope_dim]
            q = torch.cat([q_content, q_rope], dim=-1)
            # K is size [bsz, num_heads, kv_seq_len, head_dim + rope_dim]
            k = torch.cat([k_content, k_rope_expanded], dim=-1)

        # Generate Sliding Window / Causal Mask
        if self._cached_mask is None or self._cached_mask.size() != (q_len, kv_seq_len):
            q_idx = torch.arange(q_len, device=q.device).unsqueeze(1)
            kv_start_idx = kv_seq_len - q_len
            q_idx = q_idx + kv_start_idx
            kv_idx = torch.arange(kv_seq_len, device=q.device).unsqueeze(0)
            
            distance = q_idx - kv_idx
            is_future = distance < 0
            
            sliding_window = getattr(self.config, 'sliding_window_size', 4096)
            landmark_interval = getattr(self.config, 'landmark_interval', 512)
            
            if sliding_window > 0 and kv_seq_len > sliding_window:
                in_window = (distance >= 0) & (distance < sliding_window)
                is_landmark = (kv_idx % landmark_interval == 0)
                allowed = (in_window | is_landmark) & (~is_future)
            else:
                allowed = ~is_future
                
            mask = torch.zeros((q_len, kv_seq_len), device=q.device, dtype=q.dtype)
            mask = mask.masked_fill(~allowed, float("-inf"))
            self._cached_mask = mask
        else:
            mask = self._cached_mask.to(q.dtype)
        
        # Apply external attention mask if provided
        final_mask = mask.unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            final_mask = final_mask + attention_mask
            
        # Use PyTorch's optimized scaled_dot_product_attention (FlashAttention where available)
        # Note: final_mask must be boolean or float additive mask. Ours is -inf / 0.
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=final_mask,
            dropout_p=0.0,
            is_causal=False  # Mask already handles causality
        )
        
        # Reshape output back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        
        return self.o_proj(attn_output), new_past_key_value
