# precision_estimation/core/generator/relu_generator.py
import torch
from typing import Dict, Any, Tuple

class ReLUInputGenerator:


    def __init__(
        self,
        input_shape: Tuple[int, ...] = (1, 256, 32, 32),
        distribution: str = "normal",
        device: str = "cpu",
        seed: int = 2025
    ):
        self.input_shape = input_shape
        self.distribution = distribution
        self.device = torch.device(device)
        torch.manual_seed(seed)

    def generate(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.distribution == "normal":
            x = torch.randn(self.input_shape, device=self.device)
        elif self.distribution == "uniform":
            x = (torch.rand(self.input_shape, device=self.device) - 0.5) * 4.0
        elif self.distribution == "boundary":
            tiny = torch.tensor(1e-6, device=self.device)
            sign = (torch.randint(0, 2, self.input_shape, device=self.device) * 2 - 1).float()
            x = (torch.rand(self.input_shape, device=self.device) * tiny) * sign
        elif self.distribution == "adversarial_sum":
            base = torch.randn(self.input_shape, device=self.device) * 1e-3
            noise = (torch.rand_like(base) - 0.5) * 1e-7
            x = base + noise
        else:
            raise ValueError(f"Unknown distribution {self.distribution}")

        meta = dict(distribution=self.distribution)
        return x, meta

