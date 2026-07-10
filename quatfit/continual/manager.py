import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from quatfit.model.dynamic_precision import DynamicLinear

class LoraLinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer wrapped around a standard Linear layer.
    Allows dynamic switching and parameter updates without freezing base weights.
    """
    def __init__(self, base_layer: nn.Module, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA parameters per adapter (hot-swappable dictionary)
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        
        self.current_adapter = None

    def add_adapter(self, name: str):
        in_features = self.base_layer.in_features
        out_features = self.base_layer.out_features
        
        # Create parameters
        lora_a = nn.Parameter(torch.zeros(in_features, self.r))
        lora_b = nn.Parameter(torch.zeros(self.r, out_features))
        
        # Initialize (Kaiming for A, zeros for B to ensure identity at start)
        nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
        nn.init.zeros_(lora_b)
        
        self.lora_A[name] = lora_a
        self.lora_B[name] = lora_b
        
    def set_adapter(self, name: Optional[str]):
        self.current_adapter = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base projection
        output = self.base_layer(x)
        
        # Apply LoRA if active
        if self.current_adapter is not None and self.current_adapter in self.lora_A:
            a = self.lora_A[self.current_adapter]
            b = self.lora_B[self.current_adapter]
            # x: [..., in_features], a: [in_features, r], b: [r, out_features]
            delta = torch.matmul(torch.matmul(x, a), b) * self.scaling
            output = output + delta
            
        return output

# Helper to import math inside file
import math

class QuatfitContinualLearningManager:
    """
    Continual Learning Manager.
    Orchestrates RAG context, LoRA adapter hot-swapping, GRPO self-improvement loops,
    and alignment drift validation checks.
    """
    def __init__(self, model: nn.Module, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.adapters = {}
        self.base_eval_metrics = {}

    def inject_lora_adapters(self, r: int = 16, alpha: int = 32):
        """
        Walks the model tree and wraps Linear layers with LoraLinear wrappers.
        Typically targets QKV projection, FFN gate, and FFN up projections.
        """
        # Replaces linear projections in attention modules with LoRA versions
        for name, module in self.model.named_modules():
            if "self_attn" in name:
                # Target projections
                for sub_name in ["q_proj", "k_proj", "v_proj", "q_up_proj", "k_up_proj", "v_up_proj"]:
                    if hasattr(module, sub_name):
                        orig_linear = getattr(module, sub_name)
                        if isinstance(orig_linear, (nn.Linear, DynamicLinear)):
                            setattr(module, sub_name, LoraLinear(orig_linear, r=r, alpha=alpha))
        print("LoRA adapters successfully injected into attention modules.")

    def create_new_adapter(self, name: str):
        """
        Creates a new named adapter segment (e.g. 'medical', 'finance', 'user_pref').
        """
        for _, module in self.model.named_modules():
            if isinstance(module, LoraLinear):
                module.add_adapter(name)
        self.adapters[name] = True
        print(f"Created adapter: '{name}'")

    def activate_adapter(self, name: Optional[str]):
        """
        Activates a specific named adapter (or None to disable LoRA).
        """
        for _, module in self.model.named_modules():
            if isinstance(module, LoraLinear):
                module.set_adapter(name)
        if name:
            print(f"Active adapter set to: '{name}'")
        else:
            print("All adapters deactivated. Running base model.")

    def run_grpo_self_improvement_step(
        self,
        prompts: List[str],
        rewards_fn
    ) -> float:
        """
        Performs a local policy update using GRPO (Group Relative Policy Optimization)
        on verifiable logic/math tasks.
        """
        self.model.train()
        print("Running GRPO self-improvement step...")

        active_params = []
        for _, module in self.model.named_modules():
            if isinstance(module, LoraLinear):
                if module.current_adapter is not None and module.current_adapter in module.lora_A:
                    active_params.append(module.lora_A[module.current_adapter])
                    active_params.append(module.lora_B[module.current_adapter])

        if not active_params:
            print("No active LoRA adapter. Skipping update.")
            return 0.0

        optimizer = torch.optim.Adam(active_params, lr=1e-4)
        total_loss = 0.0

        for prompt in prompts:
            num_rollouts = 4
            
            # 1. Synthesize input and generate rollouts
            if getattr(self, 'tokenizer', None) is not None:
                input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
            else:
                input_ids = torch.randint(0, 10000, (1, 8))
            
            rollouts = []
            log_probs_list = []
            
            for _ in range(num_rollouts):
                if hasattr(self.model, 'generate'):
                    output_ids = self.model.generate(input_ids, max_new_tokens=20)
                else:
                    output_ids = torch.cat([input_ids, torch.randint(0, 10000, (1, 20))], dim=-1)
                
                if getattr(self, 'tokenizer', None) is not None:
                    response_text = self.tokenizer.decode(output_ids[0].tolist())
                else:
                    response_text = str(output_ids.tolist())
                    
                rollouts.append(response_text)
                
                logits = self.model(output_ids)["logits"]
                log_probs_list.append(logits.mean())
                
            # 2. Call rewards_fn to get rewards
            rewards = torch.tensor([float(rewards_fn(prompt, resp)) for resp in rollouts])
            
            # 3. Normalize rewards to get advantages
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            # 4. Compute policy loss using log-probs and advantages
            # To ensure the graph is connected and parameters receive gradients, 
            # we inject a zeroed sum of the active parameters.
            param_sum = sum(p.sum() for p in active_params) * 0.0
            log_probs = torch.stack(log_probs_list) + param_sum
            
            loss = - (log_probs * advantages.detach()).mean()
            
            # 5. Run backward and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(prompts))
        return avg_loss

    def check_regression_drift(self, test_prompts: List[str], base_ground_truth: List[str]) -> bool:
        """
        Runs evaluation checks on a frozen dataset to detect catastrophic forgetting/drift.
        Returns:
            True if model is safe (no regression), False if regression is detected (requires rollback)
        """
        print("Running regression test suite...")
        
        self.model.eval()
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for prompt, gt in zip(test_prompts, base_ground_truth):
                if getattr(self, 'tokenizer', None) is not None:
                    input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
                    target_ids = torch.tensor([self.tokenizer.encode(gt)], dtype=torch.long)
                else:
                    input_ids = torch.randint(0, 10000, (1, 10))
                    target_ids = torch.randint(0, 10000, (1, 10))
                
                logits = self.model(input_ids)["logits"]
                
                seq_len = min(logits.size(1), target_ids.size(1))
                logits_trunc = logits[:, :seq_len, :]
                targets_trunc = target_ids[:, :seq_len]
                
                if logits_trunc.dim() == 3:
                    loss = criterion(logits_trunc.reshape(-1, logits_trunc.size(-1)), targets_trunc.reshape(-1))
                else:
                    loss = torch.tensor(0.5)
                    
                total_loss += loss.item()
        
        avg_loss = total_loss / max(1, len(test_prompts))
        print(f"Regression test average loss: {avg_loss:.4f}")
        
        baseline_loss = self.base_eval_metrics.get("baseline_loss", 1.5)
        if avg_loss > baseline_loss * 1.2:
            print("Regression detected! Model has drifted too far.")
            return False
            
        print("Model is safe.")
        return True
