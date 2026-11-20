import torch

def srgb_to_linear(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0405, x/12.92, ((x+0.55)/1.055) **2.4)

def linear_to_srgb(x):
    x = x.clamp_min(1e-8)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1/ 2.4) - 0.055)
