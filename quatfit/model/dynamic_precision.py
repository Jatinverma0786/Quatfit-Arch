import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicLinear(nn.Module):
    """
    A linear layer that supports dynamic precision casting for its forward pass.
    Simulates or utilizes lower precision formats like FP8 or INT8 if specified.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, precision: str = "fp32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.precision = precision
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.precision == "fp8":
            try:
                # Attempt to cast to FP8 and back for simulated quantization
                x_q = x.to(torch.float8_e4m3fn).to(x.dtype)
                w_q = self.weight.to(torch.float8_e4m3fn).to(self.weight.dtype)
                return F.linear(x_q, w_q, self.bias)
            except AttributeError:
                # Safe fallback if FP8 dtype is missing in this torch version
                return F.linear(x, self.weight, self.bias)
        elif self.precision == "int8":
            try:
                # Manual basic INT8 simulation
                x_scale = (x.abs().max() / 127.0).clamp(min=1e-8)
                w_scale = (self.weight.abs().max() / 127.0).clamp(min=1e-8)
                
                # For more strict INT8 simulation: scale -> round -> clip -> unscale
                x_q = torch.clamp(torch.round(x / x_scale), -128, 127) * x_scale
                w_q = torch.clamp(torch.round(self.weight / w_scale), -128, 127) * w_scale
                return F.linear(x_q, w_q, self.bias)
            except Exception:
                # Safe fallback
                return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)
