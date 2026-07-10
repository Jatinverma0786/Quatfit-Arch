import torch
import torch.nn as nn
from typing import Tuple, Optional
from quatfit.model.config import QuatfitConfig
from quatfit.model.normalization import RMSNorm

class QuatfitCoTVerifier(nn.Module):
    """
    Chain-of-Thought (CoT) Verifier.
    A lightweight sub-model that evaluates the logical consistency of reasoning steps.
    """
    def __init__(self, config: QuatfitConfig):
        super().__init__()
        self.config = config
        self.base_hidden_size = config.hidden_size
        self.verifier_hidden_size = config.verifier_hidden_size
        self.num_layers = config.verifier_num_layers
        
        # Projection from base hidden size to verifier hidden size
        self.input_proj = nn.Linear(self.base_hidden_size, self.verifier_hidden_size)
        
        # Stack of standard Transformer encoder/decoder layers for verification
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.verifier_hidden_size,
            nhead=4,
            dim_feedforward=self.verifier_hidden_size * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)
        
        # Binary classification head: Correct vs. Incorrect reasoning step/token
        self.verifier_head = nn.Linear(self.verifier_hidden_size, 2)

    def segment_steps(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, step_separator_token_id: int) -> torch.Tensor:
        bsz, seq_len, dim = hidden_states.shape
        step_reps = []
        for b in range(bsz):
            sep_idx = (input_ids[b] == step_separator_token_id).nonzero(as_tuple=True)[0]
            start = 0
            b_reps = []
            for end in sep_idx:
                if end > start:
                    b_reps.append(hidden_states[b, start:end].mean(dim=0))
                else:
                    b_reps.append(torch.zeros(dim, device=hidden_states.device))
                start = end + 1
            if start < seq_len:
                b_reps.append(hidden_states[b, start:].mean(dim=0))
            if not b_reps:
                b_reps.append(hidden_states[b].mean(dim=0))
            step_reps.append(torch.stack(b_reps))
            
        max_steps = max(r.shape[0] for r in step_reps)
        padded_reps = torch.zeros((bsz, max_steps, dim), device=hidden_states.device)
        for b in range(bsz):
            padded_reps[b, :step_reps[b].shape[0]] = step_reps[b]
            
        return padded_reps

    def forward(
        self,
        base_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        step_separator_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            base_hidden_states: Hidden states from base model [batch_size, seq_len, base_hidden_size]
            attention_mask: Attention mask [batch_size, seq_len, seq_len]
        Returns:
            logits: Step-level correctness logits [batch_size, seq_len, 2]
        """
        # Project to verifier hidden size
        x = self.input_proj(base_hidden_states)
        
        if input_ids is not None and step_separator_token_id is not None:
            x = self.segment_steps(x, input_ids, step_separator_token_id)
            # update attention_mask if necessary, simplified for now
            attention_mask = None 
            
        # Apply transformer layers (using decoder, so passing self as both tgt and memory for simplicity)
        x = self.transformer(x, x, tgt_mask=attention_mask, memory_mask=attention_mask)
        
        # Project to classification logits
        logits = self.verifier_head(x)
        return logits
