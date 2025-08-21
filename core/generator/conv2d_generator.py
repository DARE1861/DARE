# precision_estimation/core/generator/conv2d_generator.py
import torch
from typing import Dict, Any, Tuple

class Conv2DInputGenerator:

    def __init__(
        self,
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
        weight_shape: Tuple[int, int, int, int] = (64, 3, 3, 3),
        distribution: str = "normal",
        device: str = "cpu",
        seed: int = 2025
    ):
        self.input_shape = input_shape
        self.weight_shape = weight_shape
        self.distribution = distribution
        self.device = torch.device(device)
        torch.manual_seed(seed)

    def generate(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.distribution == "normal":
            x = torch.randn(self.input_shape, device=self.device)
            w = torch.randn(self.weight_shape, device=self.device) * 0.1
        elif self.distribution == "uniform":
            x = (torch.rand(self.input_shape, device=self.device) - 0.5) * 2.0
            w = (torch.rand(self.weight_shape, device=self.device) - 0.5) * 0.2
        elif self.distribution == "boundary":
            tiny = torch.tensor(6.1e-5, device=self.device)
            sign = (torch.randint(0, 2, self.input_shape, device=self.device) * 2 - 1).float()
            x = (torch.rand(self.input_shape, device=self.device) * tiny) * sign + tiny * 0.5
            w = torch.randn(self.weight_shape, device=self.device) * 0.05
        elif self.distribution == "adversarial_sum":
            base = torch.randn(self.input_shape, device=self.device) * 1e-2
            noise = (torch.rand_like(base) - 0.5) * 1e-6
            x = base + noise
            w = torch.randn(self.weight_shape, device=self.device) * 0.1
        else:
            raise ValueError(f"Unknown distribution {self.distribution}")

        conv_params = dict(stride=1, padding=1, dilation=1, groups=1)
        meta = dict(distribution=self.distribution, conv_params=conv_params)
        return x, w, meta
