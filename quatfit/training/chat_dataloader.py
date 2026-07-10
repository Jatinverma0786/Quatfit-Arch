import torch
from typing import Generator, Tuple, List
from quatfit.tokenizer import SimpleBPETokenizer

class ChatStreamingDataLoader:
    """
    Streams and tokenizes conversational datasets on-the-fly.
    Formats data using <|user|> and <|assistant|> tags and creates a loss_mask
    to prevent training on user prompts.
    """
    def __init__(
        self,
        dataset_name: str = "WithinUsAI/claude_mythos_distilled_25k",
        dataset_config: str = "default",
        split: str = "train",
        max_seq_len: int = 512,
        batch_size: int = 4,
        tokenizer_path: str = "tokenizer.json"
    ):
        from datasets import load_dataset
        
        self.tokenizer = SimpleBPETokenizer.load(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        
        # Ensure our tokens exist
        self.user_token_id = self.tokenizer.encoder.get("<|user|>", self.tokenizer.encoder.get("<unk>"))
        self.asst_token_id = self.tokenizer.encoder.get("<|assistant|>", self.tokenizer.encoder.get("<unk>"))
        self.eos_token_id = self.tokenizer.encoder.get("<eos>", self.tokenizer.encoder.get("<unk>"))
        self.bos_token_id = self.tokenizer.encoder.get("<bos>", self.tokenizer.encoder.get("<unk>"))
        
        print(f"Loading conversational streaming dataset: '{dataset_name}'...")
        # Fallback to without config if it fails
        try:
            self.dataset = load_dataset(dataset_name, name=dataset_config, split=split, streaming=True)
        except Exception:
            self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        
        # Persistent iterators (lazily initialized)
        self._dataset_iter = None
        self._generator = None

    def _format_and_tokenize(self) -> Generator[Tuple[List[int], List[float]], None, None]:
        """
        Yields (token_ids, loss_mask) for each conversation.
        """
        if self._dataset_iter is None:
            self._dataset_iter = iter(self.dataset)
        for example in self._dataset_iter:
            # Datasets may have 'conversations' or 'messages'
            messages = example.get("conversations", example.get("messages", []))
            
            if not messages:
                # If it's plain text, we just treat it as an assistant response for pre-training robustness
                text = example.get("text", "")
                if text:
                    tokens = self.tokenizer.encode(text)
                    yield tokens, [1.0] * len(tokens)
                continue
                
            token_ids = [self.bos_token_id]
            loss_mask = [0.0] # mask bos
            
            for msg in messages:
                role = msg.get("from", msg.get("role", "")).lower()
                content = msg.get("value", msg.get("content", ""))
                
                if role in ["human", "user"]:
                    # User turn
                    token_ids.append(self.user_token_id)
                    loss_mask.append(0.0)
                    
                    content_tokens = self.tokenizer.encode(content)[1:-1] # skip bos/eos
                    token_ids.extend(content_tokens)
                    loss_mask.extend([0.0] * len(content_tokens)) # DO NOT train on user prompt
                    
                elif role in ["gpt", "assistant", "bot"]:
                    # Assistant turn
                    token_ids.append(self.asst_token_id)
                    loss_mask.append(1.0)
                    
                    content_tokens = self.tokenizer.encode(content)[1:-1]
                    token_ids.extend(content_tokens)
                    loss_mask.extend([1.0] * len(content_tokens)) # TRAIN on assistant response
                    
                    token_ids.append(self.eos_token_id)
                    loss_mask.append(1.0)
            
            if len(token_ids) > 1:
                yield token_ids, loss_mask

    def _packed_sequence_generator(self) -> Generator[Tuple[List[int], List[float]], None, None]:
        """
        Packs streamed tokens into fixed-length sequence blocks.
        """
        gen = self._format_and_tokenize()
        current_seq_ids = []
        current_seq_mask = []
        
        while True:
            try:
                tokens, masks = next(gen)
                current_seq_ids.extend(tokens)
                current_seq_mask.extend(masks)
                
                while len(current_seq_ids) >= self.max_seq_len:
                    yield (
                        current_seq_ids[:self.max_seq_len], 
                        current_seq_mask[:self.max_seq_len]
                    )
                    current_seq_ids = current_seq_ids[self.max_seq_len:]
                    current_seq_mask = current_seq_mask[self.max_seq_len:]
            except StopIteration:
                break

    def _ensure_generator(self):
        """Lazily initialize the persistent pack generator."""
        if self._generator is None:
            self._generator = self._packed_sequence_generator()

    def get_batch(self, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a packed batch from the streamed dataset.
        Returns: input_ids, target_ids, loss_mask
        """
        self._ensure_generator()
        batch_ids = []
        batch_masks = []
        
        for _ in range(self.batch_size):
            try:
                ids, masks = next(self._generator)
                batch_ids.append(ids)
                batch_masks.append(masks)
            except StopIteration:
                break
                
        if not batch_ids:
            raise StopIteration("Dataset stream fully exhausted.")
            
        tensor_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
        tensor_masks = torch.tensor(batch_masks, dtype=torch.float32, device=device)
        
        # Targets are shifted by 1 inside the loss function normally, 
        # but for simplicity we return identical inputs/targets and let the loop handle shift
        return tensor_ids, tensor_ids, tensor_masks
