import torch
import torch.nn as nn
from typing import Tuple, Optional
from quatfit.model.config import QuatfitConfig
from quatfit.model.ffn import SwiGLUFFN
from quatfit.model.router import AuxLossFreeRouter

class QuatfitMoELayer(nn.Module):
    """
    Mixture-of-Experts (MoE) Layer combining dynamic routing
    across specialized experts with an always-on shared expert.
    """
    def __init__(self, config: QuatfitConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.hidden_size = config.hidden_size
        
        # Router module
        self.router = AuxLossFreeRouter(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            top_k=self.top_k
        )
        
        # Routed Experts List
        self.experts = nn.ModuleList([
            SwiGLUFFN(self.hidden_size, config.expert_ffn_dim, precision=config.precision)
            for _ in range(self.num_experts)
        ])
        
        # Shared Expert (always-on for general representations)
        self.use_shared = config.num_shared_experts > 0
        if self.use_shared:
            self.shared_expert = SwiGLUFFN(self.hidden_size, config.expert_ffn_dim, precision=config.precision)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
        Returns:
            output: Aggregated output tensor of shape [batch_size, seq_len, hidden_size]
            loads: Expert load statistics for monitoring/logging
        """
        orig_shape = x.shape
        x_flat = x.view(-1, self.hidden_size)
        
        # Get expert indices, weights, and load fractions
        topk_idx, topk_weights, loads = self.router(x_flat)
        
        # Initialize output buffer for routing path
        out_flat = torch.zeros_like(x_flat)
        
        # Process active tokens for each expert
        active_experts = topk_idx.unique().tolist()
        for i in active_experts:
            mask = (topk_idx == i)
                
            token_indices, k_slots = torch.where(mask)
            expert_inputs = x_flat[token_indices]
            expert_outputs = self.experts[i](expert_inputs)
            
            # Scale outputs by matching routing weight
            weights = topk_weights[token_indices, k_slots].unsqueeze(-1)
            
            # Scatter sum back to routing output buffer
            out_flat.index_add_(0, token_indices, expert_outputs * weights)
            
        # Reshape routed output
        out = out_flat.view(orig_shape)
        
        # Add shared expert output
        if self.use_shared:
            out = out + self.shared_expert(x)
            
        return out, loads
