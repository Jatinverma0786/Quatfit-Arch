import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
from quatfit.model.normalization import RMSNorm

class AdaptiveComputationController(nn.Module):
    """
    Adaptive Computation Controller.
    Computes confidence score at layer checkpoints to decide if tokens can exit early.
    """
    def __init__(self, hidden_size: int, num_checkpoints: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_checkpoints = num_checkpoints
        
        # Norm and projection for each checkpoint
        self.norms = nn.ModuleList([RMSNorm(hidden_size) for _ in range(num_checkpoints)])
        self.classifiers = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_checkpoints)])
        
        # Default exit confidence thresholds for each checkpoint (calibrated post-training)
        self.register_buffer("thresholds", torch.ones(num_checkpoints) * 0.8)
        self.register_buffer("temperatures", torch.ones(num_checkpoints))

    def set_thresholds(self, new_thresholds: torch.Tensor):
        assert new_thresholds.shape[0] == self.num_checkpoints
        self.thresholds.copy_(new_thresholds)

    def calibrate(self, checkpoint_idx: int, logits: torch.Tensor, labels: torch.Tensor):
        """
        Calibrate the temperature for a specific checkpoint using validation data.
        Performs a grid search to minimize Negative Log Likelihood (NLL).
        """
        best_temp = 1.0
        best_loss = float('inf')
        
        # Simple grid search over temperatures from 0.1 to 5.0
        temps = torch.linspace(0.1, 5.0, steps=50, device=logits.device)
        for t in temps:
            scaled_logits = logits / t
            # Using BCE loss since it's a binary decision (exit or not)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scaled_logits, labels.float())
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_temp = t.item()
                
        self.temperatures[checkpoint_idx] = best_temp

    def forward(
        self,
        h: torch.Tensor,
        checkpoint_idx: int,
        active_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Hidden states from current layer [batch_size, seq_len, hidden_size]
            checkpoint_idx: Index of the current checkpoint (0-indexed)
            active_mask: Boolean mask of tokens still active [batch_size, seq_len]
        Returns:
            confidence: Confidence score per token [batch_size, seq_len]
            exit_mask: Boolean mask indicating which tokens should exit [batch_size, seq_len]
            new_active_mask: Boolean mask of remaining active tokens [batch_size, seq_len]
        """
        batch_size, seq_len, _ = h.shape
        
        # Apply norm and calculate raw confidence logits
        normed_h = self.norms[checkpoint_idx](h)
        logits = self.classifiers[checkpoint_idx](normed_h).squeeze(-1) # [batch_size, seq_len]
        confidence = torch.sigmoid(logits / self.temperatures[checkpoint_idx])
        
        # Determine exit decisions based on calibrated threshold
        threshold = self.thresholds[checkpoint_idx]
        
        # We only exit tokens that are currently active AND exceed the threshold
        if active_mask is not None:
            exit_mask = active_mask & (confidence >= threshold)
            new_active_mask = active_mask & (confidence < threshold)
        else:
            exit_mask = confidence >= threshold
            new_active_mask = confidence < threshold
            
        return confidence, exit_mask, new_active_mask
