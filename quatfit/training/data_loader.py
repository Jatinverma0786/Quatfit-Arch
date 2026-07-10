import torch
from typing import List, Dict, Generator, Tuple

class QuatfitPackedDataLoader:
    """
    Sequence packing dataloader.
    Concatenates documents with special separator tokens (e.g. <eos>) 
    into fixed-length blocks to prevent GPU padding waste.
    """
    def __init__(
        self,
        tokenized_corpus: List[List[int]],
        max_seq_len: int,
        batch_size: int,
        eos_token_id: int = 2
    ):
        self.corpus = tokenized_corpus
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.eos_id = eos_token_id
        
        # Pointer in the corpus list
        self.doc_ptr = 0
        
        # Persistent generator (lazily initialized)
        self._generator = None

    def _pack_generator(self) -> Generator[List[int], None, None]:
        """
        Packs documents continuously into sequences of max_seq_len.
        """
        current_seq = []
        while True:
            # Loop corpus continuously
            doc = self.corpus[self.doc_ptr]
            self.doc_ptr = (self.doc_ptr + 1) % len(self.corpus)
            
            # Add doc plus EOS separator
            current_seq.extend(doc + [self.eos_id])
            
            # Yield full chunks
            while len(current_seq) >= self.max_seq_len:
                yield current_seq[:self.max_seq_len]
                current_seq = current_seq[self.max_seq_len:]

    def _ensure_generator(self):
        """Lazily initialize the persistent pack generator."""
        if self._generator is None:
            self._generator = self._pack_generator()

    def get_batch(self, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assembles a batch of packed tokens.
        Returns:
            inputs: [batch_size, seq_len]
            targets: [batch_size, seq_len] (same, targets are shifted inside loss)
        """
        self._ensure_generator()
        batch_tokens = []
        
        for _ in range(self.batch_size):
            batch_tokens.append(next(self._generator))
            
        tensor_batch = torch.tensor(batch_tokens, dtype=torch.long, device=device)
        return tensor_batch, tensor_batch
