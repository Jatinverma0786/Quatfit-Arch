import torch
import torch.nn as nn
from typing import Tuple, Optional
from quatfit.model.config import QuatfitConfig
from quatfit.model.normalization import RMSNorm
from quatfit.model.attention import QuatfitAttention
from quatfit.model.ffn import SwiGLUFFN
from quatfit.model.moe_layer import QuatfitMoELayer

class QuatfitTransformerBlock(nn.Module):
    """
    A single Quatfit Transformer Block.
    Integrates Pre-norm RMSNorm, GQA/MLA Attention, and Dense or MoE FFN layer.
    """
    def __init__(self, config: QuatfitConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.is_moe = config.is_moe and (layer_idx >= config.num_dense_layers)
        self.layer_dropout_prob = config.layer_dropout_prob
        
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.norm_epsilon)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.norm_epsilon)
        
        self.self_attn = QuatfitAttention(config, layer_idx=layer_idx)
        
        if self.is_moe:
            self.moe = QuatfitMoELayer(config)
        else:
            dense_dim = config.dense_ffn_dim if config.dense_ffn_dim is not None else config.expert_ffn_dim
            self.dense_ffn = SwiGLUFFN(self.hidden_size, dense_dim, precision=config.precision)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_size]
            position_ids: Positional indices [batch, seq_len]
        Returns:
            hidden_states: Updated hidden states [batch, seq_len, hidden_size]
            new_past_key_value: Cached keys/values
            loads: Expert load fractions (if MoE layer)
        """
        # Layer dropout for adaptive computation (only drop MoE blocks to preserve dense backbone)
        if self.is_moe and self.training and torch.rand(1).item() < self.layer_dropout_prob:
            # Skip computation, return residual connection with unchanged KV cache
            return hidden_states, past_key_value, None
            
        # Attention block (Pre-norm)
        normed_hidden = self.input_layernorm(hidden_states)
        attn_outputs, new_past_key_value = self.self_attn(
            hidden_states=normed_hidden,
            position_ids=position_ids,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            use_cache=use_cache
        )
        hidden_states = hidden_states + attn_outputs
        
        # FFN block (Pre-norm)
        normed_hidden = self.post_attention_layernorm(hidden_states)
        
        loads = None
        if self.is_moe:
            ffn_outputs, loads = self.moe(normed_hidden)
        else:
            ffn_outputs = self.dense_ffn(normed_hidden)
            
        hidden_states = hidden_states + ffn_outputs
        
        return hidden_states, new_past_key_value, loads
