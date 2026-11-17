from utils.ray_utils import ray_length_factor
import math
from types import SimpleNamespace
import torch

def test_ray_factor():
    cam = SimpleNamespace(FoVx=math.radians(90), FoVy=2 * math.atan(math.tan(math.radians(45)) * 360 / 640))
    rho = ray_length_factor(cam, 360, 640, device="cpu")
    assert rho.shape == (1, 360, 640)
    assert abs(rho[0, 179:181, 319:321].mean().item() - 1.0) < 1e-4
    corner = math.sqrt(1 + math.tan(cam.FoVx / 2) ** 2 + math.tan(cam.FoVy / 2) ** 2)
    assert abs(rho[0, 0, 0].item() - corner) < 0.01
