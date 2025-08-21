# precision_estimation/core/generator/matmul_generator.py
import torch
from typing import Dict, Any, Tuple, Optional


class MatmulInputGenerator:

    def __init__(
        self,
        a_shape: Tuple[int, int] = (512, 256),
        b_shape: Tuple[int, int] = (256, 512),
        distribution: str = "normal",
        device: str = "cpu",
        seed: Optional[int] = 2025,
    ):
        assert a_shape[1] == b_shape[0], "Matmul dim mismatch: a_shape[1] must equal b_shape[0]"
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.distribution = distribution
        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(int(seed))

    def generate(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.distribution == "normal":
            A = torch.randn(self.a_shape, device=self.device)
            B = torch.randn(self.b_shape, device=self.device) * 0.1
        elif self.distribution == "uniform":
            A = (torch.rand(self.a_shape, device=self.device) - 0.5) * 2.0
            B = (torch.rand(self.b_shape, device=self.device) - 0.5) * 0.2
        elif self.distribution == "boundary":
            tiny = torch.tensor(6.1e-5, device=self.device)
            sign = (torch.randint(0, 2, self.a_shape, device=self.device) * 2 - 1).float()
            A = (torch.rand(self.a_shape, device=self.device) * tiny) * sign + tiny * 0.5
            B = torch.randn(self.b_shape, device=self.device) * 0.05
        elif self.distribution == "adversarial_sum":
            base = torch.randn(self.a_shape, device=self.device) * 1e-2
            noise = (torch.rand_like(base) - 0.5) * 1e-6
            A = base + noise
            B = torch.randn(self.b_shape, device=self.device) * 0.1
        else:
            raise ValueError(f"Unknown distribution {self.distribution}")

        meta = dict(distribution=self.distribution, matmul_params={"a_shape": self.a_shape, "b_shape": self.b_shape})
        return A, B, meta
