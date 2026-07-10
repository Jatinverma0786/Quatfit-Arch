import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from quatfit.model.quatfit_model import QuatfitModel

class QuatfitSpeculativeDecoder:
    """
    Self-Speculative Decoder.
    Generates draft tokens using early network layers (exits),
    then validates them using the full depth in a single batch pass.
    """
    def __init__(self, model: QuatfitModel, draft_checkpoint_idx: int = 0):
        self.model = model
        self.draft_chk = draft_checkpoint_idx
        self.checkpoints = model.config.adaptive_exit_checkpoints
        
        # Verify checkpoint index is valid
        assert self.draft_chk < len(self.checkpoints), "Invalid draft checkpoint index"
        self.draft_exit_layer = self.checkpoints[self.draft_chk]

    def _generate_draft_step(self, input_ids: torch.Tensor, past_key_values: Optional[List] = None) -> Tuple[torch.Tensor, List]:
        """
        Executes a fast forward pass terminating early at the draft checkpoint.
        """
        # Save original exit checkpoints
        orig_checkpoints = self.model.checkpoints
        orig_use_adaptive = self.model.use_adaptive
        
        try:
            self.model.checkpoints = [self.draft_exit_layer]
            self.model.use_adaptive = True
            self.model.adaptive_controller.thresholds.fill_(0.0)
            
            with torch.no_grad():
                outputs = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
        finally:
            self.model.checkpoints = orig_checkpoints
            self.model.use_adaptive = orig_use_adaptive
            self.model.adaptive_controller.thresholds.copy_(orig_thresholds)
        
        # Get next token prediction from the early exit
        logits = outputs["logits"][:, -1, :] # [batch, vocab_size]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        return next_token, outputs["past_key_values"]

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 20, draft_steps: int = 4) -> torch.Tensor:
        """
        Generates text using self-speculative decoding.
        Args:
            input_ids: Prompt token IDs [batch_size, seq_len]
            max_new_tokens: Total tokens to generate
            draft_steps: Number of tokens to draft before verification
        """
        # Fallback to standard autoregressive generation if model has no adaptive checkpoints (e.g. Nano)
        if not hasattr(self.model, "adaptive_controller") or not self.model.config.use_adaptive_computation:
            current_seq = input_ids
            past_key_values = None
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    outputs = self.model(current_seq, past_key_values=past_key_values, use_cache=True)
                logits = outputs["logits"][:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                current_seq = torch.cat([current_seq, next_token], dim=-1)
                past_key_values = outputs["past_key_values"]
            return current_seq
            
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Pre-fill phase (full model pass)
        with torch.no_grad():
            outputs = self.model(input_ids[:, :-1], use_cache=True)
            
        past_key_values = outputs["past_key_values"]
        unprocessed_tokens = input_ids[:, -1:]
        current_seq = input_ids
        
        tokens_generated = 0
        while tokens_generated < max_new_tokens:
            # 1. Draft Phase: Generate K candidate tokens using early exit
            draft_tokens = []
            draft_pkv = past_key_values
            
            # Process the unprocessed token to get the first draft token
            next_draft, draft_pkv = self._generate_draft_step(unprocessed_tokens, past_key_values=draft_pkv)
            draft_tokens.append(next_draft)
            draft_input = next_draft
            
            for _ in range(draft_steps - 1):
                next_draft, draft_pkv = self._generate_draft_step(draft_input, past_key_values=draft_pkv)
                draft_tokens.append(next_draft)
                draft_input = next_draft
                
            draft_seq = torch.cat(draft_tokens, dim=-1) # [batch, draft_steps]
            
            # 2. Verification Phase: Single parallel forward pass through full model
            verify_input = torch.cat([unprocessed_tokens, draft_seq], dim=-1)
            
            # Disable early exits to force full verification
            orig_use_adaptive = self.model.use_adaptive
            try:
                self.model.use_adaptive = False
                with torch.no_grad():
                    verify_outputs = self.model(verify_input, past_key_values=past_key_values, use_cache=True)
            finally:
                self.model.use_adaptive = orig_use_adaptive
            
            # Get full model prediction logits for the draft slots
            verify_logits = verify_outputs["logits"] # [batch, 1 + draft_steps, vocab]
            verify_tokens = torch.argmax(verify_logits, dim=-1) # [batch, 1 + draft_steps]
            
            # Check how many draft tokens match the verification tokens
            accepted_count = 0
            for i in range(draft_steps):
                if (draft_seq[:, i] == verify_tokens[:, i]).all():
                    accepted_count += 1
                else:
                    break
                    
            # 3. Accept/Reject and append
            # Append accepted tokens, plus the verification token at the first mismatch
            accepted_tokens = draft_seq[:, :accepted_count]
            correction_token = verify_tokens[:, accepted_count:accepted_count+1]
            
            new_tokens = torch.cat([accepted_tokens, correction_token], dim=-1)
            current_seq = torch.cat([current_seq, new_tokens], dim=-1)
            tokens_generated += new_tokens.size(1)
            
            # The unprocessed token for the next iteration is the correction token
            unprocessed_tokens = correction_token
            
            # Trim KV Cache to only include unprocessed_tokens and accepted_tokens
            base_length = past_key_values[0][0].shape[-2] if past_key_values else 0
            keep_length = base_length + 1 + accepted_count
            
            trimmed_pkv = []
            for layer_pkv in verify_outputs["past_key_values"]:
                trimmed_layer = tuple(t[..., :keep_length, :] if t is not None else None for t in layer_pkv)
                trimmed_pkv.append(trimmed_layer)
                
            past_key_values = trimmed_pkv
            
        return current_seq
