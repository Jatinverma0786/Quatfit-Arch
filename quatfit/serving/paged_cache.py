import torch
from typing import List

class PagedKVCacheManager:
    def __init__(self, num_blocks: int, num_heads: int, block_size: int, head_dim: int, device="cpu", dtype=torch.float32):
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = head_dim
        
        self.key_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim, device=device, dtype=dtype)
        self.value_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim, device=device, dtype=dtype)
        
        self.free_blocks = list(range(num_blocks))
        self.block_tables = [] # List of block indices for each sequence in the batch
        self.context_lengths = [] # List of sequence lengths for each sequence in the batch

    def allocate(self, seq_len: int) -> List[int]:
        num_required_blocks = (seq_len + self.block_size - 1) // self.block_size
        if num_required_blocks > len(self.free_blocks):
            raise RuntimeError("Out of memory in PagedKVCacheManager")
        
        allocated = self.free_blocks[:num_required_blocks]
        self.free_blocks = self.free_blocks[num_required_blocks:]
        return allocated
