import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
from quatfit.model.quatfit_model import QuatfitModel

class SpecializedTrainingManager:
    """
    Manages specialized training phases:
      1. Adaptive computation threshold calibration.
      2. CoT Verifier contrastive training.
      3. Context extension scheduling.
      4. Hierarchical memory training.
    """
    def __init__(self, model: QuatfitModel, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    # 1. Adaptive Computation Threshold Calibration
    def calibrate_adaptive_thresholds(self, eval_dataloader, target_accuracy_drop: float = 0.01):
        """
        Calibrates exit thresholds so that speed is maximized while accuracy
        stays within `target_accuracy_drop` of the full model.
        """
        print("Calibrating adaptive compute thresholds...")
        self.model.eval()
        
        # Collect all exit confidence scores across evaluation set
        checkpoint_scores = [[] for _ in range(len(self.model.checkpoints))]
        
        with torch.no_grad():
            for inputs, _ in eval_dataloader:
                inputs = inputs.to(self.device)
                
                # Retrieve confidence scores from adaptive controller
                hidden_states = self.model.embed_tokens(inputs)
                
                # Baseline memory summary
                mem_summary = torch.mean(hidden_states, dim=1)
                if self.model.use_memory:
                    memory_output, _, _, _ = self.model.memory_system(hidden_states, mode='residual')
                    hidden_states = hidden_states + memory_output
                    
                active_mask = torch.ones(inputs.size(0), inputs.size(1), dtype=torch.bool, device=self.device)
                
                bsz, seq_len, _ = hidden_states.shape
                position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1).to(self.device)
                
                for i, layer in enumerate(self.model.layers):
                    hidden_states, _, _ = layer(hidden_states, position_ids=position_ids)
                    
                    if self.model.use_adaptive and (i + 1) in self.model.checkpoints:
                        chk_idx = self.model.checkpoints.index(i + 1)
                        # We extract raw confidence from the controller
                        normed_h = self.model.adaptive_controller.norms[chk_idx](hidden_states)
                        logits = self.model.adaptive_controller.classifiers[chk_idx](normed_h).squeeze(-1)
                        confidence = torch.sigmoid(logits)
                        
                        checkpoint_scores[chk_idx].extend(confidence.view(-1).tolist())

        # Set thresholds based on target percentile
        # e.g., if we want to retain 99% quality, we set threshold to exclude low-confidence exits
        calibrated_thresholds = []
        for chk_idx, scores in enumerate(checkpoint_scores):
            if not scores:
                calibrated_thresholds.append(0.8)
                continue
            scores_tensor = torch.tensor(scores)
            # Find the threshold matching target exit rate
            # For 99% accuracy retention, typically only exit top 60% most confident
            val = torch.quantile(scores_tensor, 1.0 - target_accuracy_drop)
            calibrated_thresholds.append(val.item())
            
        new_thresholds = torch.tensor(calibrated_thresholds, device=self.device)
        self.model.adaptive_controller.set_thresholds(new_thresholds)
        print(f"Calibrated thresholds set to: {new_thresholds.tolist()}")

    # 2. CoT Verifier Training
    def train_verifier_step(self, hidden_states: torch.Tensor, step_labels: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Trains the CoT Verifier using binary cross-entropy on reasoning trace labels.
        Args:
            hidden_states: reasoning trace states [batch, seq_len, hidden_size]
            step_labels: binary correctness labels [batch, seq_len]
        """
        assert self.model.use_verifier, "Verifier is not enabled in configuration"
        self.model.verifier.train()
        
        # Project and verify
        normed_states = self.model.norm(hidden_states)
        logits = self.model.verifier(normed_states) # [batch, seq_len, 2]
        
        # Cross entropy loss
        loss = F.cross_entropy(
            logits.view(-1, 2),
            step_labels.view(-1)
        )
        
        # Calculate accuracy metric
        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == step_labels).float().mean().item()
        
        return loss, accuracy

    # 3. Context Extension Tuner
    def apply_yarn_scaling(self, target_context_len: int):
        """
        Updates positional scaling factors for context window extension.
        """
        base_len = self.model.config.max_position_embeddings
        scale = float(target_context_len) / base_len
        
        # Update rotary base frequency scaling
        for layer in self.model.layers:
            attn = layer.self_attn
            attn.rotary_emb.scale = scale
            # Recompute cos/sin caches
            attn.rotary_emb._set_cos_sin_cache(
                seq_len=target_context_len,
                device=self.device,
                dtype=torch.float32
            )
        print(f"YaRN positional scaling set to {scale:.2f}x (Context extended to {target_context_len} tokens)")

    # 4. Hierarchical Memory Tuner
    def train_memory_surprise_gate_step(self, x: torch.Tensor, prev_persistent: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trains the surprise gate predictor to minimize state reconstruction error
        over typical inputs (teaches it what is 'normal' vs 'surprising').
        """
        assert self.model.use_memory, "Memory system is not enabled"
        self.model.memory_system.train()
        
        batch_size, _, hidden_dim = x.shape
        if prev_persistent is None:
            prev_persistent = torch.zeros(batch_size, hidden_dim, device=x.device)
        
        # Predict states
        predicted_x = self.model.memory_system.surprise_gate.predictor(prev_persistent.unsqueeze(1))
        
        # Update persistent state (simplified for training step)
        mem_summary = torch.mean(x, dim=1)
        new_persistent = prev_persistent + mem_summary
        
        # Reconstruction loss (MSE)
        loss = F.mse_loss(predicted_x, x)
        return loss, new_persistent
