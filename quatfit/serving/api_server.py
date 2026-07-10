import json
import torch
from typing import Dict, Any, List, Optional
from quatfit.model.quatfit_model import QuatfitModel
from quatfit.serving.paged_cache import PagedKVCacheManager

class QuatfitAPIServer:
    """
    OpenAI-compatible local server wrapper for Quatfit.
    Manages multi-model routing (Nano vs Base) and KV cache sessions.
    """
    def __init__(self, nano_model: QuatfitModel, base_model: QuatfitModel, tokenizer):
        self.nano = nano_model
        self.base = base_model
        self.tokenizer = tokenizer
        self.sessions = {} # session_id -> KV caches

    def route_request(self, prompt: str, domain: str = "general") -> QuatfitModel:
        """
        Multi-Model Request Routing:
        Routes queries based on length, complexity, and specific domain tags.
        """
        # Complex coding/math/reasoning tasks go to Base
        if domain in ["math", "code", "science"] or len(prompt) > 2048:
            print("Routing request to Quatfit Base (complex domain/long sequence)...")
            return self.base
        else:
            # Simple chat/factual queries go to Nano
            print("Routing request to Quatfit Nano (lightweight chat)...")
            return self.nano

    def handle_chat_completion(self, request_json: str) -> str:
        """
        Processes standard OpenAI-compatible completions request.
        """
        data = json.loads(request_json)
        messages = data.get("messages", [])
        
        # Simple concat for prompt
        prompt = " ".join([m.get("content", "") for m in messages])
        domain = data.get("domain", "general")
        
        # Decide which model to run
        model = self.route_request(prompt, domain=domain)
        
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        
        # Create a dummy cache manager
        dummy_cache = PagedKVCacheManager(
            num_blocks=16, 
            num_heads=model.config.num_heads if hasattr(model, 'config') else 8, 
            block_size=16, 
            head_dim=64
        )
        
        # Generate new tokens
        output_ids = model.generate(input_ids, max_new_tokens=50, past_key_values=dummy_cache)
        response_text = self.tokenizer.decode(output_ids[0].tolist())
        
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "quatfit-base" if model == self.base else "quatfit-nano",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 10,
                "total_tokens": len(prompt.split()) + 10
            }
        }
        
        return json.dumps(mock_response)
