import torch
from typing import List

class PagedKVCacheManager:
    def __init__(self, num_blocks: int, num_heads: int, block_size: int, head_dim: int, config=None, device="cpu", dtype=torch.float32):
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = head_dim
        self.num_kv_heads = getattr(config, 'num_key_value_groups', num_heads) if config else num_heads
        
        if config and getattr(config, 'attention_type', 'gqa') == 'mla':
            # For MLA, KV is compressed into a single vector (effectively 1 KV head per sequence)
            self.num_kv_heads = 1
            key_dim = getattr(config, 'mla_kv_compression_dim', head_dim)
            value_dim = getattr(config, 'mla_rope_head_dim', head_dim)
            self.key_cache = torch.zeros(num_blocks, self.num_kv_heads, block_size, key_dim, device=device, dtype=dtype)
            self.value_cache = torch.zeros(num_blocks, self.num_kv_heads, block_size, value_dim, device=device, dtype=dtype)
        else:
            self.key_cache = torch.zeros(num_blocks, self.num_kv_heads, block_size, head_dim, device=device, dtype=dtype)
            self.value_cache = torch.zeros(num_blocks, self.num_kv_heads, block_size, head_dim, device=device, dtype=dtype)
        
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

    def grow(self, seq_idx: int, additional_tokens: int):
        num_required_blocks = (additional_tokens + self.block_size - 1) // self.block_size
        if num_required_blocks > len(self.free_blocks):
            raise RuntimeError("Out of memory in PagedKVCacheManager")
        
        allocated = self.free_blocks[:num_required_blocks]
        self.free_blocks = self.free_blocks[num_required_blocks:]
        self.block_tables[seq_idx].extend(allocated)
