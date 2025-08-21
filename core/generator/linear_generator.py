# precision_estimation/core/generator/linear_generator.py
import torch
from typing import Dict, Any, Tuple

class LinearInputGenerator:


    def __init__(
        self,
        input_shape: Tuple[int, ...] = (32, 1024),  # (batch_size, in_features)
        weight_shape: Tuple[int, int] = (4096, 1024),  # (out_features, in_features)
        bias: bool = False,
        distribution: str = "normal",
        device: str = "cpu",
        seed: int = 2025
    ):
        self.input_shape = input_shape
        self.weight_shape = weight_shape
        self.bias = bias
        self.distribution = distribution
        self.device = torch.device(device)
        torch.manual_seed(seed)

    def generate(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.distribution == "normal":
            x = torch.randn(self.input_shape, device=self.device)
            w = torch.randn(self.weight_shape, device=self.device) * 0.1
            b = torch.randn(self.weight_shape[0], device=self.device) * 0.01 if self.bias else None
        elif self.distribution == "uniform":
            x = (torch.rand(self.input_shape, device=self.device) - 0.5) * 2.0
            w = (torch.rand(self.weight_shape, device=self.device) - 0.5) * 0.2
            b = (torch.rand(self.weight_shape[0], device=self.device) - 0.5) * 0.02 if self.bias else None
        elif self.distribution == "boundary":
            tiny = torch.tensor(1e-4, device=self.device)
            sign = (torch.randint(0, 2, self.input_shape, device=self.device) * 2 - 1).float()
            x = (torch.rand(self.input_shape, device=self.device) * tiny) * sign
            w = torch.randn(self.weight_shape, device=self.device) * 0.05
            b = torch.randn(self.weight_shape[0], device=self.device) * 0.001 if self.bias else None
        elif self.distribution == "adversarial_sum":
            base_x = torch.randn(self.input_shape, device=self.device) * 1e-2
            noise_x = (torch.rand_like(base_x) - 0.5) * 1e-6
            x = base_x + noise_x
            w = torch.randn(self.weight_shape, device=self.device) * 0.1
            b = torch.randn(self.weight_shape[0], device=self.device) * 0.01 if self.bias else None
        else:
            raise ValueError(f"Unknown distribution {self.distribution}")

        meta = dict(
            distribution=self.distribution,
            bias=self.bias,
            batch_size=self.input_shape[0] if len(self.input_shape) >= 2 else 1,
            in_features=self.weight_shape[1],
            out_features=self.weight_shape[0]
        )
        return x, w, b, meta

