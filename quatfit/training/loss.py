import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional

class QuatfitLoss(nn.Module):
    """
    Computes unified Quatfit losses:
      1. Primary Next-Token CrossEntropy.
      2. Multi-Token Prediction (MTP) loss.
      3. Checkpoint early-exit losses.
    """
    def __init__(self, mtp_loss_weight: float = 0.3, exit_loss_weight: float = 0.1):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn_none = nn.CrossEntropyLoss(reduction='none')
        self.mtp_weight = mtp_loss_weight
        self.exit_weight = exit_loss_weight

    def _masked_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes cross-entropy loss with optional masking.
        Args:
            logits: [N, vocab_size] or [batch, seq, vocab_size]
            targets: [N] or [batch, seq]
            mask: Optional [N] or [batch, seq] float mask (1.0=include, 0.0=exclude)
        """
        if mask is None:
            return self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        per_token_loss = self.loss_fn_none(
            logits.view(-1, logits.size(-1)), targets.view(-1)
        )  # [N]
        flat_mask = mask.view(-1)  # [N]
        masked_loss = per_token_loss * flat_mask
        num_active = flat_mask.sum().clamp(min=1.0)
        return masked_loss.sum() / num_active

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs: Outputs dict from QuatfitModel forward pass
            targets: Target token IDs of shape [batch_size, seq_len]
            loss_mask: Optional float tensor of shape [batch_size, seq_len] with
                       1.0 for positions to include and 0.0 for positions to mask out
        Returns:
            loss: Scaled combined loss tensor
            metrics: Loggable metrics dictionary
        """
        # Targets shifted by 1 for standard next-token prediction
        # inputs are [b, seq_len], targets should match output logits shape [b, seq_len-1]
        logits = outputs["logits"] # [batch_size, seq_len, vocab_size]
        
        # Shift inputs and targets:
        # standard LM predicts t+1 given t
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets[..., 1:].contiguous()
        
        # Shift mask to align with shifted logits/targets
        shift_mask = None
        if loss_mask is not None:
            shift_mask = loss_mask[..., 1:].contiguous()
        
        # 1. Primary Loss
        primary_loss = self._masked_loss(shift_logits, shift_targets, shift_mask)
        
        combined_loss = primary_loss
        metrics = {"primary_loss": primary_loss.item()}
        
        # 2. Checkpoint Early-Exit Losses
        checkpoint_logits = outputs.get("checkpoint_logits", [])
        if checkpoint_logits:
            exit_losses = []
            for idx, chk_logit in enumerate(checkpoint_logits):
                chk_shift_logits = chk_logit[..., :-1, :].contiguous()
                chk_loss = self._masked_loss(chk_shift_logits, shift_targets, shift_mask)
                exit_losses.append(chk_loss)
                metrics[f"exit_loss_chk_{idx}"] = chk_loss.item()
                
            mean_exit_loss = torch.stack(exit_losses).mean()
            combined_loss = combined_loss + self.exit_weight * mean_exit_loss
            metrics["mean_exit_loss"] = mean_exit_loss.item()

        # 3. Multi-Token Prediction (MTP) Loss
        mtp_logits_list = outputs.get("mtp_logits")
        if mtp_logits_list is not None:
            mtp_losses = []
            for k, mtp_logits in enumerate(mtp_logits_list):
                # k-th head predicts token t + (k + 1)
                # Head 0 predicts t + 1 (already covered by primary loss, but deepseek trains it separately too)
                # Let's align targets: target for t is token t + (k + 1)
                # So we shift logits and targets by k + 1
                shift_k = k + 1
                mtp_shift_logits = mtp_logits[..., :-shift_k, :].contiguous()
                mtp_shift_targets = targets[..., shift_k:].contiguous()
                
                # Shift mask for MTP alignment
                mtp_mask = None
                if loss_mask is not None:
                    mtp_mask = loss_mask[..., shift_k:].contiguous()
                
                if mtp_shift_targets.numel() > 0:
                    mtp_loss_k = self._masked_loss(
                        mtp_shift_logits, mtp_shift_targets, mtp_mask
                    )
                    mtp_losses.append(mtp_loss_k)
                    metrics[f"mtp_loss_k_{k}"] = mtp_loss_k.item()
            
            if mtp_losses:
                mean_mtp_loss = torch.stack(mtp_losses).mean()
                combined_loss = combined_loss + self.mtp_weight * mean_mtp_loss
                metrics["mean_mtp_loss"] = mean_mtp_loss.item()

        metrics["total_loss"] = combined_loss.item()
        return combined_loss, metrics
