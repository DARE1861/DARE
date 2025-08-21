# precision_estimation/core/generator/softmax_generator.py
import torch
from typing import Dict, Any, Tuple

class SoftmaxInputGenerator:

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (8, 512),
        distribution: str = "normal",
        device: str = "cpu",
        seed: int = 2025,
        temperature: float = 1.0,
        dim: int = -1,
    ):
        self.input_shape = input_shape
        self.distribution = distribution
        self.device = torch.device(device)
        self.temperature = temperature
        self.dim = dim
        torch.manual_seed(seed)

    def generate(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.distribution == "normal":
            x = torch.randn(self.input_shape, device=self.device)
        elif self.distribution == "uniform":
            x = (torch.rand(self.input_shape, device=self.device) - 0.5) * 4.0
        elif self.distribution == "boundary":
            boundary_val = 10.0
            sign = (torch.randint(0, 2, self.input_shape, device=self.device) * 2 - 1).float()
            x = torch.rand(self.input_shape, device=self.device) * boundary_val * sign
        elif self.distribution == "large_logits":
            x = torch.randn(self.input_shape, device=self.device) * 5.0
            mask = torch.rand(self.input_shape, device=self.device) < 0.1
            x[mask] = x[mask] + torch.rand_like(x[mask]) * 20.0
        elif self.distribution == "adversarial_sum":
            base = torch.randn(self.input_shape, device=self.device) * 0.1
            noise = (torch.rand_like(base) - 0.5) * 1e-5
            x = base + noise
        elif self.distribution == "temperature_scaled":
            x = torch.randn(self.input_shape, device=self.device) / self.temperature
        else:
            raise ValueError(f"Unknown distribution {self.distribution}")

        meta = dict(
            distribution=self.distribution,
            temperature=self.temperature,
            dim=self.dim
        )
        return x, meta
