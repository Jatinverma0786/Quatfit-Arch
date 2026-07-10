import torch
from quatfit.model.config import QuatfitConfig
from quatfit.model.moe_layer import QuatfitMoELayer

config = QuatfitConfig.get_preset_config('mini')
# The nano uses dense, mini uses moe
layer = QuatfitMoELayer(config)

x = torch.randn(2, 8, config.hidden_size)
print("Forwarding MoE layer...")
out, loads = layer(x)
print("Done!")
