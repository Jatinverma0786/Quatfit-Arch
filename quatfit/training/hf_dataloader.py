import torch
from typing import Generator, Tuple, Optional, List
from quatfit.tokenizer import SimpleBPETokenizer

class HuggingFaceStreamingDataLoader:
    """
    Streams and tokenizes high-quality HuggingFace datasets on-the-fly,
    using sequence packing to eliminate padding tokens.
    
    Default dataset: "HuggingFaceFW/fineweb-edu" (Highest quality Web Edu tokens).
    """
    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        dataset_config: str = "sample-10BT", # use a sample config for faster streaming startup
        split: str = "train",
        max_seq_len: int = 128,
        batch_size: int = 4,
        tokenizer_path: str = "tokenizer.json"
    ):
        # Dynamically import HuggingFace datasets to avoid strict dependencies at startup
        from datasets import load_dataset
        
        self.tokenizer = SimpleBPETokenizer.load(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        
        print(f"Loading HuggingFace streaming dataset: '{dataset_name}' ({dataset_config})...")
        self.dataset = load_dataset(dataset_name, name=dataset_config, split=split, streaming=True)
        
        # Persistent iterators (lazily initialized)
        self._dataset_iter = None
        self._generator = None

    def _token_generator(self) -> Generator[int, None, None]:
        """
        Yields tokens one by one from the streamed dataset.
        """
        if self._dataset_iter is None:
            self._dataset_iter = iter(self.dataset)
        for example in self._dataset_iter:
            # Fineweb-edu uses the 'text' field for document content
            text = example.get("text", "")
            if text:
                token_ids = self.tokenizer.encode(text)
                for token_id in token_ids:
                    yield token_id

    def _packed_sequence_generator(self) -> Generator[List[int], None, None]:
        """
        Packs streamed tokens into fixed-length sequence blocks.
        """
        token_gen = self._token_generator()
        current_seq = []
        
        while True:
            try:
                token = next(token_gen)
                current_seq.append(token)
                
                if len(current_seq) >= self.max_seq_len:
                    yield current_seq[:self.max_seq_len]
                    current_seq = current_seq[self.max_seq_len:]
            except StopIteration:
                # Dataset exhausted
                break

    def _ensure_generator(self):
        """Lazily initialize the persistent pack generator."""
        if self._generator is None:
            self._generator = self._packed_sequence_generator()

    def get_batch(self, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a packed batch from the streamed dataset.
        """
        self._ensure_generator()
        batch_tokens = []
        
        for _ in range(self.batch_size):
            try:
                batch_tokens.append(next(self._generator))
            except StopIteration:
                # Fallback in case dataset stream is exhausted
                break
                
        if not batch_tokens:
            raise StopIteration("HuggingFace dataset stream fully exhausted.")
            
        tensor_batch = torch.tensor(batch_tokens, dtype=torch.long, device=device)
        return tensor_batch, tensor_batch
