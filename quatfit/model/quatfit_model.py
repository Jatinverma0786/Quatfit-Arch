import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict, Any
from quatfit.model.config import QuatfitConfig
from quatfit.model.embedding import FactoredEmbedding
from quatfit.model.transformer_block import QuatfitTransformerBlock
from quatfit.model.adaptive_controller import AdaptiveComputationController
from quatfit.model.memory import QuatfitHierarchicalMemory
from quatfit.model.cot_verifier import QuatfitCoTVerifier
from quatfit.model.output_head import QuatfitOutputHead
from quatfit.model.normalization import RMSNorm

class QuatfitModel(nn.Module):
    """
    Unified Quatfit Model Architecture.
    Assembles dense/MoE layers, adaptive computation exits, hierarchical memory,
    CoT verifier, and the output projection/MTP head.
    """
    def __init__(self, config: QuatfitConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.use_adaptive = config.use_adaptive_computation
        self.checkpoints = config.adaptive_exit_checkpoints
        
        # Token embedding
        self.embed_tokens = FactoredEmbedding(config)
        
        # Shared embedding weight parameter for tying weights with output head
        shared_weight = None
        if config.use_factored_embeddings:
            shared_weight = self.embed_tokens.emb_low.weight
        else:
            shared_weight = self.embed_tokens.emb_full.weight
            
        # Transformer Blocks Stack
        self.layers = nn.ModuleList([
            QuatfitTransformerBlock(config, layer_idx=i)
            for i in range(self.num_layers)
        ])
        
        # Normalization before output head
        self.norm = RMSNorm(self.hidden_size, eps=config.norm_epsilon)
        
        # Output head supporting LM and MTP
        self.output_head = QuatfitOutputHead(config, shared_embedding_weight=shared_weight)
        
        # Hierarchical Memory System
        self.use_memory = config.use_hierarchical_memory
        if self.use_memory:
            self.memory_system = QuatfitHierarchicalMemory(config)
            
        # Adaptive Computation Controller
        if self.use_adaptive:
            self.adaptive_controller = AdaptiveComputationController(
                hidden_size=self.hidden_size,
                num_checkpoints=len(self.checkpoints)
            )
            
        # Chain-of-Thought Verifier
        self.use_verifier = config.use_cot_verifier
        if self.use_verifier:
            self.verifier = QuatfitCoTVerifier(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        # Hierarchical memory inputs
        prev_persistent: Optional[torch.Tensor] = None,
        archive_keys: Optional[torch.Tensor] = None,
        archive_values: Optional[torch.Tensor] = None,
        return_verifier: bool = False,
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids: Token indices [batch_size, seq_len]
            position_ids: Positional indices [batch_size, seq_len]
        Returns:
            dict containing:
                logits: Next-token logits [batch_size, seq_len, vocab_size]
                mtp_logits: List of K future token predictions (during training/MTP evaluation)
                checkpoint_logits: List of logits at exited checkpoints (during training)
                past_key_values: Cached keys/values
                new_persistent: Updated persistent memory
                new_archive_keys, new_archive_values: Updated archive memory
                verifier_logits: CoT verification scores (if requested)
        """
        bsz, seq_len = input_ids.shape
        
        if position_ids is None:
            past_length = past_key_values[0][0].shape[-2] if past_key_values is not None and past_key_values[0] is not None else 0
            position_ids = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
            
        # 1. Embed input tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Initialize outputs
        final_logits = torch.zeros(bsz, seq_len, self.config.vocab_size, device=input_ids.device, dtype=hidden_states.dtype)
        checkpoint_logits = []
        mtp_logits = None
        new_past_key_values = [] if use_cache else None
        
        # Track active tokens (for early-exit routing during inference)
        active_mask = torch.ones(bsz, seq_len, dtype=torch.bool, device=input_ids.device)
        
        # Hierarchical memory integration
        new_persistent = prev_persistent
        new_keys = archive_keys
        new_values = archive_values
        if self.use_memory:
            memory_output, new_persistent, new_keys, new_values = self.memory_system(
                x=hidden_states,
                prev_persistent=prev_persistent,
                archive_keys=archive_keys,
                archive_values=archive_values,
                mode='residual'
            )
            hidden_states = hidden_states + memory_output
            
        # Process layer-by-layer
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            # Forward pass through block
            hidden_states, new_layer_past, loads = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                past_key_value=layer_past,
                attention_mask=attention_mask,
                use_cache=use_cache
            )
            
            if use_cache:
                new_past_key_values.append(new_layer_past)
                
            # Adaptive Computation Early-Exit Checkpoints
            if self.use_adaptive and (i + 1) in self.checkpoints:
                chk_idx = self.checkpoints.index(i + 1)
                
                # Check confidence for early exit
                confidence, exit_mask, new_active = self.adaptive_controller(
                    h=hidden_states,
                    checkpoint_idx=chk_idx,
                    active_mask=active_mask
                )
                
                if self.training:
                    # In training mode, we do NOT skip layers (to ensure gradient flow for all params),
                    # but we project checkpoints to compute multi-exit loss.
                    normed_chk = self.norm(hidden_states)
                    chk_logits, _ = self.output_head(normed_chk)
                    checkpoint_logits.append(chk_logits)
                else:
                    # In inference mode, we project exiting tokens and terminate computation for them
                    if exit_mask.any():
                        # Extract exiting token hidden states, run through norm + output head
                        # For simple implementation, we write to final_logits
                        normed_chk = self.norm(hidden_states)
                        chk_logits, _ = self.output_head(normed_chk)
                        
                        # Write logits for exiting tokens only
                        final_logits[exit_mask] = chk_logits[exit_mask]
                        
                    # Update active mask
                    active_mask = new_active
                    
                    # If all tokens have exited, break early!
                    if not active_mask.any():
                        break

        # Final projection layer
        normed_hidden = self.norm(hidden_states)
        primary_logits, mtp_logits = self.output_head(normed_hidden, return_mtp=self.training)
        
        if self.training:
            # Training mode: final logits are the final layer projection
            final_logits = primary_logits
        else:
            # Inference mode: write remaining active tokens to final_logits
            if active_mask.any():
                final_logits[active_mask] = primary_logits[active_mask]
                
        # Optional Chain-of-Thought verifier
        verifier_logits = None
        if self.use_verifier and return_verifier:
            verifier_logits = self.verifier(normed_hidden)
            
        return {
            "logits": final_logits,
            "mtp_logits": mtp_logits,
            "checkpoint_logits": checkpoint_logits,
            "past_key_values": new_past_key_values,
            "new_persistent": new_persistent,
            "new_archive_keys": new_keys,
            "new_archive_values": new_values,
            "verifier_logits": verifier_logits
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True
    ) -> torch.Tensor:
        curr_input_ids = input_ids
        
        past_key_values = None
        prev_persistent = None
        archive_keys = None
        archive_values = None
        
        for _ in range(max_new_tokens):
            outputs = self.forward(
                input_ids=curr_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                prev_persistent=prev_persistent,
                archive_keys=archive_keys,
                archive_values=archive_values
            )
            
            past_key_values = outputs.get("past_key_values")
            prev_persistent = outputs.get("new_persistent")
            archive_keys = outputs.get("new_archive_keys")
            archive_values = outputs.get("new_archive_values")
            
            next_token_logits = outputs["logits"][:, -1, :]
            
            if do_sample:
                next_token_logits = next_token_logits / max(temperature, 1e-5)
                
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                    
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            curr_input_ids = next_token
            
        return input_ids
