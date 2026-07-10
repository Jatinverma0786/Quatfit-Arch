import json
import os
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional

class SimpleBPETokenizer:
    """
    Lightweight, pure-Python Byte-Pair Encoding (BPE) tokenizer.
    Ensures zero-dependency reliability for local training and inference.
    """
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>", "<|user|>", "<|assistant|>"]
        
        self.encoder = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.decoder = {idx: token for idx, token in enumerate(self.special_tokens)}
        self.merges = {}

    def train(self, corpus: str):
        print("Training BPE tokenizer on corpus...")
        # Clean and split corpus into words
        words = re.findall(r"\w+|[^\w\s]", corpus, re.UNICODE)
        
        # Word counts
        word_counts = Counter(words)
        
        # Initialize vocab with characters
        chars = set("".join(words))
        for char in chars:
            if char not in self.encoder:
                idx = len(self.encoder)
                self.encoder[char] = idx
                self.decoder[idx] = char
                
        # Represent words as lists of characters
        splits = {word: list(word) for word in word_counts}
        
        # Iterative BPE merges
        while len(self.encoder) < self.vocab_size:
            # Count adjacent pairs
            pair_counts = Counter()
            for word, count in word_counts.items():
                split = splits[word]
                for i in range(len(split) - 1):
                    pair_counts[(split[i], split[i+1])] += count
                    
            if not pair_counts:
                break
                
            # Get most frequent pair
            best_pair, max_count = pair_counts.most_common(1)[0]
            if max_count < 2:
                break
                
            # Perform merge
            new_token = "".join(best_pair)
            new_idx = len(self.encoder)
            self.encoder[new_token] = new_idx
            self.decoder[new_idx] = new_token
            self.merges[best_pair] = new_token
            
            # Update word splits
            for word in word_counts:
                split = splits[word]
                i = 0
                while i < len(split) - 1:
                    if (split[i], split[i+1]) == best_pair:
                        split[i:i+2] = [new_token]
                    else:
                        i += 1
                        
        # Pad vocabulary with dummy unused tokens to match target vocab_size
        while len(self.encoder) < self.vocab_size:
            dummy_token = f"<unused_{len(self.encoder)}>"
            idx = len(self.encoder)
            self.encoder[dummy_token] = idx
            self.decoder[idx] = dummy_token
            
        print(f"BPE Tokenizer trained successfully. Vocab size: {len(self.encoder)}")

    def encode(self, text: str) -> List[int]:
        if text in self.encoder:
            return [self.encoder[text]]
            
        words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        token_ids = [self.encoder["<bos>"]]
        
        if not hasattr(self, 'merge_ranks'):
            self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges.keys())}
            
        for word in words:
            tokens = list(word)
            while len(tokens) > 1:
                best_pair = None
                best_rank = float('inf')
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i+1])
                    if pair in self.merge_ranks:
                        rank = self.merge_ranks[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = i
                if best_pair is None:
                    break
                merged = tokens[best_pair] + tokens[best_pair + 1]
                tokens = tokens[:best_pair] + [merged] + tokens[best_pair+2:]
                
            for t in tokens:
                token_ids.append(self.encoder.get(t, self.encoder["<unk>"]))
                        
        token_ids.append(self.encoder["<eos>"])
        return token_ids

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for idx in ids:
            token = self.decoder.get(idx, "<unk>")
            if token not in self.special_tokens:
                tokens.append(token)
        # Join directly, BPE parts shouldn't have spaces injected everywhere
        text = "".join(tokens)
        return text

    def save(self, path: str):
        data = {
            "vocab_size": self.vocab_size,
            "encoder": self.encoder,
            "merges": {f"{k[0]}\t{k[1]}": v for k, v in self.merges.items()}
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "SimpleBPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.encoder = data["encoder"]
        tokenizer.decoder = {int(v): k for k, v in data["encoder"].items()}
        tokenizer.merges = {tuple(k.split("\t")): v for k, v in data["merges"].items()}
        return tokenizer
