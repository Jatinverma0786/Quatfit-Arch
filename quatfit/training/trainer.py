import os
import math
import torch
import torch.nn as nn
from typing import Dict, Any, List
from quatfit.model.quatfit_model import QuatfitModel
from quatfit.training.loss import QuatfitLoss
from quatfit.training.data_loader import QuatfitPackedDataLoader

class QuatfitTrainer:
    """
    Unified trainer manager for Quatfit models.
    Coordinates training steps, curriculum context length updates, 
    loss computations, and optimization.
    """
    def __init__(
        self,
        model: QuatfitModel,
        dataloader: QuatfitPackedDataLoader,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        loss_weight_mtp: float = 0.3,
        loss_weight_exit: float = 0.1,
        device: str = "cpu",
        gradient_accumulation_steps: int = 1,
        total_steps: int = 10000,
        warmup_steps: int = 500
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        
        # Gradient accumulation
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._micro_step = 0
        
        # Mixed-precision training
        self.use_amp = (device != "cpu")
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        
        # Loss function
        self.loss_fn = QuatfitLoss(
            mtp_loss_weight=loss_weight_mtp,
            exit_loss_weight=loss_weight_exit
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler: linear warmup + cosine annealing
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self._base_lr = learning_rate
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_lambda
        )

    def _lr_lambda(self, current_step: int) -> float:
        """Linear warmup then cosine decay schedule."""
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        progress = float(current_step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def train_step(self) -> Dict[str, float]:
        """
        Executes a single forward + backward + optimization step.
        Supports mixed-precision and gradient accumulation.
        """
        self.model.train()
        
        # Retrieve batch
        inputs, targets = self.dataloader.get_batch(device=self.device)
        
        # Forward pass with mixed-precision
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            outputs = self.model(inputs)
            loss, metrics = self.loss_fn(outputs, targets)
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass through scaler
        self.scaler.scale(loss).backward()
        
        self._micro_step += 1
        
        # Only step optimizer every N micro-steps
        if self._micro_step % self.gradient_accumulation_steps == 0:
            # Unscale before clipping
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step through scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Step LR scheduler
            self.scheduler.step()
        
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        return metrics

    def update_curriculum(self, current_step: int, total_steps: int):
        """
        Curriculum Learning context extension:
        Gradually increase context window as training progresses.
        """
        # Example schedule: start at 4K context, ramp up to max base (e.g. 32K)
        # For POC, let's keep it bounded
        base_max = self.model.config.max_position_embeddings
        if current_step < int(total_steps * 0.5):
            self.dataloader.max_seq_len = min(4096, base_max)
        elif current_step < int(total_steps * 0.8):
            self.dataloader.max_seq_len = min(8192, base_max)
        else:
            self.dataloader.max_seq_len = base_max

    def save_checkpoint(self, path: str, step: int):
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from {path} at step {checkpoint['step']}")
        return checkpoint["step"]
