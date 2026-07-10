import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from quatfit.model.config import QuatfitConfig

class QuatfitOutputHead(nn.Module):
    """
    Unified Output Head supporting primary Next-Token Prediction
    and auxiliary Multi-Token Prediction (MTP) for speculative decoding.
    """
    def __init__(self, config: QuatfitConfig, shared_embedding_weight: Optional[nn.Parameter] = None):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.use_mtp = config.use_mtp
        self.mtp_depth = config.mtp_depth
        
        # Primary Language Model head
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        if shared_embedding_weight is not None:
            if self.lm_head.weight.shape == shared_embedding_weight.shape:
                self.lm_head.weight = shared_embedding_weight
            else:
                import logging
                logging.warning(f"Skipping weight tying: LM head shape {self.lm_head.weight.shape} != embed shape {shared_embedding_weight.shape}")
            
        # Multi-Token Prediction (MTP) Heads
        if self.use_mtp:
            # Each MTP head predicts token at t + k (where k is 1 to mtp_depth)
            self.mtp_heads = nn.ModuleList([
                nn.Linear(self.hidden_size, self.vocab_size, bias=False)
                for _ in range(self.mtp_depth)
            ])
            # Optional weight sharing/initialization
            for head in self.mtp_heads:
                if shared_embedding_weight is not None:
                    if head.weight.shape == shared_embedding_weight.shape:
                        head.weight.data.copy_(shared_embedding_weight.data)

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_mtp: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            hidden_states: Hidden states from current layer/exit [batch_size, seq_len, hidden_size]
            return_mtp: Whether to return MTP predictions
        Returns:
            primary_logits: Logits for next token [batch_size, seq_len, vocab_size]
            mtp_logits: List of K tensors [batch_size, seq_len, vocab_size] for future predictions
        """
        # Primary next-token prediction
        primary_logits = self.lm_head(hidden_states)
        
        mtp_logits = None
        if self.use_mtp and (return_mtp or self.training):
            mtp_logits = []
            for k in range(self.mtp_depth):
                # Project current hidden states through k-th MTP head
                logits_k = self.mtp_heads[k](hidden_states)
                mtp_logits.append(logits_k)
                
        return primary_logits, mtp_logits
