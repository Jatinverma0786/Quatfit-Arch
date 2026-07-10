import torch
import torch.nn as nn
import torch.nn.functional as F
from quatfit.model.dynamic_precision import DynamicLinear

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, precision: str = "fp32"):
        super().__init__()
        # w1: gate projection, w3: up projection, w2: down projection
        self.w1 = DynamicLinear(hidden_size, intermediate_size, bias=False, precision=precision)
        self.w3 = DynamicLinear(hidden_size, intermediate_size, bias=False, precision=precision)
        self.w2 = DynamicLinear(intermediate_size, hidden_size, bias=False, precision=precision)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # silu is the same as swish: x * sigmoid(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
