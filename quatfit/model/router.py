import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class AuxLossFreeRouter(nn.Module):
    """
    Auxiliary-Loss-Free MoE Router.
    Dynamically adjusts expert biases to achieve load balance without degrading main loss.
    """
    def __init__(self, hidden_size: int, num_experts: int, top_k: int, bias_update_rate: float = 0.01):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias_update_rate = bias_update_rate
        
        # Router weight matrix
        self.weight = nn.Parameter(torch.randn(hidden_size, num_experts) * 0.02)
        
        # Load balancing bias term for each expert (not updated by gradients)
        self.register_buffer("expert_bias", torch.zeros(num_experts))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tokens of shape [batch_size * seq_len, hidden_size]
        Returns:
            topk_idx: Top-k expert indices per token [batch_size * seq_len, top_k]
            topk_weights: Normalized routing weights [batch_size * seq_len, top_k]
            load: Fraction of tokens routed to each expert [num_experts]
        """
        # Calculate raw gating logits
        logits = torch.matmul(x, self.weight) # [N, num_experts]
        
        # Add dynamic bias for load balancing (detached from gradient graph)
        gating_scores = logits + self.expert_bias.detach()
        
        # Inject Gumbel noise during training for expert exploration
        if self.training:
            noise = -torch.empty_like(gating_scores).exponential_().log()
            gating_scores = gating_scores + 0.1 * noise
        
        # Select top-k experts
        topk_scores, topk_idx = torch.topk(gating_scores, self.top_k, dim=-1)
        
        # Softmax over top-k scores to get normalized routing weights
        topk_weights = F.softmax(topk_scores, dim=-1)
        
        # Calculate expert loads for this batch (for statistics and bias updates)
        # count how many times each expert was selected in the top-k
        flat_topk_idx = topk_idx.view(-1)
        expert_counts = torch.bincount(flat_topk_idx, minlength=self.num_experts).float()
        
        total_tokens = x.size(0) * self.top_k
        actual_fractions = expert_counts / total_tokens
        
        # Update biases during training mode
        if self.training:
            target_fraction = 1.0 / self.num_experts
            # If actual fraction > target, decrease bias (reduce routing affinity)
            # If actual fraction < target, increase bias (boost routing affinity)
            load_diff = actual_fractions - target_fraction
            self.expert_bias.sub_(self.bias_update_rate * load_diff)
            # Clamp the bias to prevent explosion
            max_bias = getattr(self, 'max_bias', 10.0)
            self.expert_bias.clamp_(-max_bias, max_bias)
            # Note: In DistributedDataParallel (DDP), expert_bias may become desynced across GPUs
            # since it's updated manually on each worker without an all-reduce operation.
            
        return topk_idx, topk_weights, actual_fractions
