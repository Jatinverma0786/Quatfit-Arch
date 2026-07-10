import torch
import torch.nn as nn
from quatfit.model.config import QuatfitConfig

class FactoredEmbedding(nn.Module):
    def __init__(self, config: QuatfitConfig):
        super().__init__()
        self.use_factored = config.use_factored_embeddings
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.factored_embed_dim = config.factored_embed_dim

        if self.use_factored:
            self.emb_low = nn.Embedding(self.vocab_size, self.factored_embed_dim)
            self.emb_proj = nn.Linear(self.factored_embed_dim, self.hidden_size, bias=False)
        else:
            self.emb_full = nn.Embedding(self.vocab_size, self.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.use_factored:
            x = self.emb_low(input_ids)
            return self.emb_proj(x)
        else:
            return self.emb_full(input_ids)
