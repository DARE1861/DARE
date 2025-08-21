# precision_estimation/core/generator/batchnorm_generator.py
import torch
from typing import Dict, Any, Tuple

class BatchNormInputGenerator:


    def __init__(
        self,
        input_shape: Tuple[int, int, int, int] = (8, 64, 32, 32),  # (N, C, H, W)
        distribution: str = "normal",
        device: str = "cpu",
        seed: int = 2025,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True
    ):
        self.input_shape = input_shape
        self.distribution = distribution
        self.device = torch.device(device)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        torch.manual_seed(seed)

    def generate(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        返回: x, weight, bias, running_mean, running_var, meta
        """
        N, C, H, W = self.input_shape
        
        if self.distribution == "normal":
            x = torch.randn(self.input_shape, device=self.device)
        elif self.distribution == "uniform":
            x = (torch.rand(self.input_shape, device=self.device) - 0.5) * 4.0
        elif self.distribution == "boundary":
            base_var = max(self.eps * 10, 1e-4)
            x = torch.randn(self.input_shape, device=self.device) * torch.sqrt(torch.tensor(base_var))
            x += torch.randn(C, device=self.device).view(1, C, 1, 1) * 1e-6
        elif self.distribution == "adversarial_sum":
            base = torch.randn(self.input_shape, device=self.device) * 0.1
            noise = (torch.rand_like(base) - 0.5) * 1e-7
            x = base + noise
        elif self.distribution == "small_batch":
            x = torch.randn(self.input_shape, device=self.device)
            if N > 2:
                x = x[:2]
                self.input_shape = (2,) + self.input_shape[1:]
        elif self.distribution == "high_variance":
            x = torch.randn(self.input_shape, device=self.device) * 100.0
            x[0, :, 0, 0] *= 1000
        else:
            raise ValueError(f"Unknown distribution {self.distribution}")

        if self.affine:
            weight = torch.ones(C, device=self.device) + torch.randn(C, device=self.device) * 0.1
            bias = torch.randn(C, device=self.device) * 0.1
        else:
            weight = None
            bias = None

        running_mean = torch.zeros(C, device=self.device)
        running_var = torch.ones(C, device=self.device)

        bn_params = dict(
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=True,
            training=True
        )
        meta = dict(
            distribution=self.distribution,
            bn_params=bn_params,
            input_shape=x.shape
        )
        
        return x, weight, bias, running_mean, running_var, meta

