import math
import torch

_cache = {}

def ray_length_factor(cam, H, W, device="cuda"):
    key = (round(float(cam.FoVx), 9), round(float(cam.FoVy), 9), int(H), int(W), str(device))
    if key not in _cache:
        tx, ty = math.tan(cam.FoVx / 2), math.tan(cam.FoVy / 2)
        xs = ((torch.arange(W, dtype=torch.float32)+0.5)/W*2 - 1) * tx
        ys = ((torch.arange(H, dtype=torch.float32)+0.5)/H*2 - 1) * ty
        y, x = torch.meshgrid(ys, xs, indexing="ij")
        _cache[key] = torch.sqrt(1+x*x+y*y)[None].to(device)
    return _cache[key]
