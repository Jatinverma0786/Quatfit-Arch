import torch
import torch.nn as nn
from typing import Dict, Any, List
from quatfit.model.quatfit_model import QuatfitModel

class QuatfitExporter:
    """
    Handles exporting Quatfit models to quantized and compressed formats,
    specifically focusing on GGUF compatible parameter extracts
    and active-parameters-only export for edge MoE.
    """
    def __init__(self, model: QuatfitModel):
        self.model = model

    def export_active_parameters_only(self, active_expert_indices: List[int], output_path: str):
        """
        Active-Parameters-Only Export:
        Reduces MoE disk size for specific deployments by exporting only 
        the dense layers + selected active expert sub-networks, removing unneeded experts.
        """
        print(f"Exporting active-parameters-only configuration for experts: {active_expert_indices}")
        
        state_dict = self.model.state_dict()
        filtered_state_dict = {}
        
        # Iterate over all parameter names in state dict
        for name, param in state_dict.items():
            # Check if parameter belongs to MoE expert
            if ".moe.experts." in name:
                # Find expert index in parameter name
                # format: layers.X.moe.experts.Y.w1.weight
                parts = name.split(".")
                expert_idx_str = parts[parts.index("experts") + 1]
                expert_idx = int(expert_idx_str)
                
                # Only keep parameter if it is in our active list
                if expert_idx in active_expert_indices:
                    filtered_state_dict[name] = param
            else:
                # Keep all non-expert weights (dense, embeddings, attention, norms)
                filtered_state_dict[name] = param

        # Save compressed state dict
        torch.save(filtered_state_dict, output_path)
        print(f"Compressed MoE checkpoint successfully written to: {output_path}")

    def export_quantized(self, output_path: str):
        """
        Exports model weights to a quantized format using fake INT8 quantization.
        """
        print(f"Exporting quantized model to {output_path}...")
        
        state_dict = self.model.state_dict()
        quantized_dict = {}
        
        for name, w in state_dict.items():
            if w.dim() >= 2: # heuristic for Linear weights
                scale = w.abs().max() / 127.0
                q_w = torch.round(w / scale).to(torch.int8)
                quantized_dict[name] = q_w
                quantized_dict[name + "_scale"] = scale
            else:
                quantized_dict[name] = w
                
        torch.save(quantized_dict, output_path)
        print("Quantized model export complete.")
