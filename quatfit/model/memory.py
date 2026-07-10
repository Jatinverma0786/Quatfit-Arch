import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from quatfit.model.config import QuatfitConfig

class SurpriseGate(nn.Module):
    """
    Surprise Gate evaluates the information novelty of incoming states
    and routes them to different tiers of the hierarchical memory.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        # A small prediction network to predict the next state from current memory
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )

    def forward(self, x: torch.Tensor, memory_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Current token hidden states [batch_size, seq_len, hidden_size]
            memory_state: Current summary of memory [batch_size, hidden_size]
        Returns:
            surprise_score: Novelty score per token [batch_size, seq_len]
        """
        # Predict the token state based on memory
        # memory_state shape: [batch, hidden_size] -> [batch, 1, hidden_size]
        predicted_x = self.predictor(memory_state.unsqueeze(1)) # [batch, 1, hidden_size]
        
        # Calculate surprise as MSE between actual and predicted state
        surprise = torch.mean((x - predicted_x) ** 2, dim=-1) # [batch, seq_len]
        return surprise


class PersistentMemory(nn.Module):
    """
    Tier 2 Memory: Fast-weight associative memory updated during inference
    using surprise-gated recurrent updates.
    """
    def __init__(self, hidden_size: int, memory_dim: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_dim = memory_dim
        
        # Gating projections
        self.update_gate = nn.Linear(hidden_size + memory_dim, memory_dim)
        self.input_proj = nn.Linear(hidden_size, memory_dim)
        self.output_proj = nn.Linear(memory_dim, hidden_size)

    def forward(self, x: torch.Tensor, prev_state: torch.Tensor, surprise: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input states [batch_size, seq_len, hidden_size]
            prev_state: Previous memory state [batch_size, memory_dim]
            surprise: Surprise score [batch_size, seq_len]
            threshold: Novelty threshold to trigger update
        Returns:
            retrieved: Retrieved representations [batch_size, seq_len, hidden_size]
            new_state: Updated memory state [batch_size, memory_dim]
        """
        batch_size, seq_len, _ = x.shape
        state = prev_state # [batch, memory_dim]
        
        # Batch processing: Process all tokens at once.
        # Compute candidates and gates for all tokens simultaneously
        candidates = torch.tanh(self.input_proj(x)) # [batch, seq_len, memory_dim]
        
        # Expand state to compute gates for each token.
        state_expanded = state.unsqueeze(1).expand(-1, seq_len, -1) # [batch, seq_len, memory_dim]
        gate_inputs = torch.cat([x, state_expanded], dim=-1)
        g = torch.sigmoid(self.update_gate(gate_inputs)) # [batch, seq_len, memory_dim]
        
        # Surprise mask
        mask = (surprise > threshold).float().unsqueeze(-1) # [batch, seq_len, 1]
        
        # Combine weighted candidates
        weighted_candidates = candidates * (mask * g) # [batch, seq_len, memory_dim]
        
        # Approximate sequential update by sum
        total_update = torch.sum(weighted_candidates, dim=1) # [batch, memory_dim]
        retain_factor = 1.0 - torch.mean(mask * g, dim=1)
        
        new_state = retain_factor * state + total_update
            
        # Retrieve information: query memory state with input
        # simple projection back to hidden size
        retrieved = self.output_proj(new_state).unsqueeze(1).expand(-1, seq_len, -1)
        return retrieved, new_state


class ArchiveMemory(nn.Module):
    """
    Tier 3 Memory: Quantized, compressed external buffer with kNN similarity search.
    """
    def __init__(self, hidden_size: int, max_archive_size: int = 10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_archive_size = max_archive_size
        
        # Projections to create compact keys/values for the archive
        self.key_proj = nn.Linear(hidden_size, hidden_size // 4, bias=False)
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        archive_keys: Optional[torch.Tensor],
        archive_values: Optional[torch.Tensor],
        k: int = 5
    ) -> torch.Tensor:
        """
        Args:
            q: Queries [batch_size, seq_len, hidden_size]
            archive_keys: Archived keys [batch_size, num_archived, key_dim]
            archive_values: Archived values [batch_size, num_archived, hidden_size]
            k: Top-k elements to retrieve
        Returns:
            retrieved: Retrieved representations [batch_size, seq_len, hidden_size]
        """
        if archive_keys is None or archive_keys.size(1) == 0:
            return torch.zeros_like(q)
            
        batch_size, seq_len, _ = q.shape
        q_key = self.key_proj(q) # [batch, seq_len, key_dim]
        
        # Calculate cosine similarity: [batch, seq_len, num_archived]
        q_norm = F.normalize(q_key, p=2, dim=-1)
        k_norm = F.normalize(archive_keys, p=2, dim=-1)
        
        # TODO: Integrate FAISS for production instead of exact kNN
        sims = []
        chunk_size = 1024
        for i in range(0, k_norm.size(1), chunk_size):
            chunk_k_norm = k_norm[:, i:i+chunk_size, :]
            sim_chunk = torch.matmul(q_norm, chunk_k_norm.transpose(-1, -2))
            sims.append(sim_chunk)
        sim = torch.cat(sims, dim=-1)
        
        # Select top-k matches
        topk_sim, topk_idx = torch.topk(sim, min(k, sim.size(-1)), dim=-1)
        
        # Softmax over top-k similarities to get weights
        weights = F.softmax(topk_sim, dim=-1).unsqueeze(-1) # [batch, seq_len, k, 1]
        
        # Gather matching values
        # archive_values: [batch, num_archived, hidden_size]
        # we need to gather values along the num_archived dimension for each query token
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size) # [batch, seq_len, k, hidden_size]
        
        # Expand archive_values to match gather structure
        expanded_values = archive_values.unsqueeze(1).expand(-1, seq_len, -1, -1)
        matched_values = torch.gather(expanded_values, dim=2, index=gather_idx) # [batch, seq_len, k, hidden_size]
        
        # Weighted sum of retrieved values
        retrieved = torch.sum(matched_values * weights, dim=2)
        return retrieved


class QuatfitHierarchicalMemory(nn.Module):
    """
    Unified Hierarchical Memory System.
    Links Working, Persistent, and Archive memory with surprise gating.
    """
    def __init__(self, config: QuatfitConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.active_tiers = config.memory_tiers_active
        
        self.surprise_gate = SurpriseGate(self.hidden_size)
        
        if 2 in self.active_tiers:
            self.persistent_memory = PersistentMemory(self.hidden_size)
        if 3 in self.active_tiers:
            self.archive_memory = ArchiveMemory(self.hidden_size)
            
        # Learned integration weights for blending memory outputs
        self.weights = nn.Parameter(torch.ones(3) / 3.0)

    def forward(
        self,
        x: torch.Tensor,
        prev_persistent: Optional[torch.Tensor] = None,
        archive_keys: Optional[torch.Tensor] = None,
        archive_values: Optional[torch.Tensor] = None,
        mode: str = 'preprocess'
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: Input representations [batch_size, seq_len, hidden_size]
            prev_persistent: Previous persistent memory state [batch_size, memory_dim]
            archive_keys: Existing archive keys [batch_size, num_archived, key_dim]
            archive_values: Existing archive values [batch_size, num_archived, hidden_size]
        Returns:
            output: Unified representation with memory context [batch_size, seq_len, hidden_size]
            new_persistent: Updated persistent memory state [batch_size, memory_dim]
            new_archive_keys: Updated archive keys [batch_size, num_archived_new, key_dim]
            new_archive_values: Updated archive values [batch_size, num_archived_new, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Tier 1: Working Memory is the input representation itself
        working_out = x
        
        # Initialize memory summaries
        persistent_out = torch.zeros_like(x)
        archive_out = torch.zeros_like(x)
        
        new_persistent = prev_persistent
        new_keys = archive_keys
        new_values = archive_values
        
        # Get memory state representation for surprise gating (mean of input if no memory state)
        if prev_persistent is not None:
            # project memory state back to hidden size if needed (we use projection in module)
            mem_summary = self.persistent_memory.output_proj(prev_persistent)
        else:
            mem_summary = torch.mean(x, dim=1) # baseline fallback [batch, hidden_size]
            
        # Calculate surprise
        surprise = self.surprise_gate(x, mem_summary)
        
        # Tier 2: Persistent Context
        if 2 in self.active_tiers:
            if prev_persistent is None:
                # Initialize state
                prev_persistent = torch.zeros(batch_size, self.persistent_memory.memory_dim, device=x.device, dtype=x.dtype)
                
            persistent_out, new_persistent = self.persistent_memory(
                x, prev_persistent, surprise, self.config.surprise_threshold_persistent
            )
            
        # Tier 3: Archive Memory
        if 3 in self.active_tiers:
            # Retrieve from archive
            archive_out = self.archive_memory(x, archive_keys, archive_values)
            
            # Archive new tokens that exceed threshold
            # Find tokens with high surprise
            high_surprise_mask = (surprise > self.config.surprise_threshold_archive)
            
            # Project tokens to key/value dimensions
            keys = self.archive_memory.key_proj(x)
            values = self.archive_memory.value_proj(x)
            
            max_size = getattr(self.config, 'max_archive_size', 10000)
            batch_new_k, batch_new_v = [], []
            max_len = 0
            
            for b in range(batch_size):
                mask = high_surprise_mask[b]
                if mask.any():
                    new_b_keys = keys[b, mask]
                    new_b_values = values[b, mask]
                    
                    # concatenate with old keys if existing
                    if archive_keys is not None:
                        new_b_keys = torch.cat([archive_keys[b], new_b_keys], dim=0)
                        new_b_values = torch.cat([archive_values[b], new_b_values], dim=0)
                        
                    if new_b_keys.size(0) > max_size:
                        new_b_keys = new_b_keys[-max_size:]
                        new_b_values = new_b_values[-max_size:]
                else:
                    new_b_keys = archive_keys[b] if archive_keys is not None else keys.new_zeros(0, keys.size(-1))
                    new_b_values = archive_values[b] if archive_values is not None else values.new_zeros(0, values.size(-1))
                    
                batch_new_k.append(new_b_keys)
                batch_new_v.append(new_b_values)
                max_len = max(max_len, new_b_keys.size(0))
            
            # Pad to max_len
            stacked_keys = torch.zeros(batch_size, max_len, keys.size(-1), device=x.device, dtype=x.dtype)
            stacked_values = torch.zeros(batch_size, max_len, values.size(-1), device=x.device, dtype=x.dtype)
            for b in range(batch_size):
                k_len = batch_new_k[b].size(0)
                if k_len > 0:
                    stacked_keys[b, -k_len:] = batch_new_k[b]
                    stacked_values[b, -k_len:] = batch_new_v[b]
            
            new_keys = stacked_keys
            new_values = stacked_values
                
        # Normalize integration weights
        norm_weights = F.softmax(self.weights, dim=0)
        
        # Blend outputs
        output = (
            norm_weights[0] * working_out +
            norm_weights[1] * persistent_out +
            norm_weights[2] * archive_out
        )
        
        if mode == 'residual':
            output = (
                norm_weights[1] * persistent_out +
                norm_weights[2] * archive_out
            )
        
        return output, new_persistent, new_keys, new_values
