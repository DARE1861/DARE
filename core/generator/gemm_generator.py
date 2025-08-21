# precision_estimation/core/generator/gemm_generator.py
import torch
from typing import Dict, Any, Tuple

class GEMMInputGenerator:


    def __init__(
        self,
        input_shape: Tuple[int, int] = (128, 512),
        weight_shape: Tuple[int, int] = (512, 256),
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

            tiny = torch.tensor(1e-6, device=self.device)
            sign = (torch.randint(0, 2, self.input_shape, device=self.device) * 2 - 1).float()
            x = (torch.rand(self.input_shape, device=self.device) * tiny) * sign + tiny * 0.1
            w = torch.randn(self.weight_shape, device=self.device) * 0.05
        elif self.distribution == "adversarial_sum":

            base = torch.randn(self.input_shape, device=self.device) * 1e-3
            noise = (torch.rand_like(base) - 0.5) * 1e-7
            x = base + noise
            w = torch.randn(self.weight_shape, device=self.device) * 0.1
        elif self.distribution == "ill_conditioned":
            U, _, V = torch.svd(torch.randn(*self.input_shape, device=self.device))
            s = torch.logspace(-6, 0, min(self.input_shape), device=self.device)
            S = torch.zeros(self.input_shape, device=self.device)
            S[:len(s), :len(s)] = torch.diag(s)
            x = U @ S @ V.T
            w = torch.randn(self.weight_shape, device=self.device) * 0.1
        else:
            raise ValueError(f"Unknown distribution {self.distribution}")

        gemm_params = dict(transpose_a=False, transpose_b=False)
        meta = dict(distribution=self.distribution, gemm_params=gemm_params)
        return x, w, meta

