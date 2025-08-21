# precision_estimation/core/generator/layernorm_generator.py
import torch
from typing import Dict, Any, Tuple

class LayerNormInputGenerator:

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (8, 512, 768), 
        normalized_shape: Tuple[int, ...] = (768,), 
        distribution: str = "normal",
        device: str = "cpu",
        seed: int = 2025,
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        self.input_shape = input_shape
        self.normalized_shape = normalized_shape
        self.distribution = distribution
        self.device = torch.device(device)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        torch.manual_seed(seed)

    def generate(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.distribution == "normal":
            x = torch.randn(self.input_shape, device=self.device)
        elif self.distribution == "uniform":
            x = (torch.rand(self.input_shape, device=self.device) - 0.5) * 4.0
        elif self.distribution == "boundary":
            x = torch.randn(self.input_shape, device=self.device) * math.sqrt(self.eps * 10)
        elif self.distribution == "adversarial_sum":
            base_mean = 100.0
            x = torch.randn(self.input_shape, device=self.device) * 0.01 + base_mean
        elif self.distribution == "small_variance":
            x = torch.randn(self.input_shape, device=self.device) * 1e-4
        else:
            raise ValueError(f"Unknown distribution {self.distribution}")

        if self.elementwise_affine:
            weight = torch.ones(self.normalized_shape, device=self.device)
            bias = torch.zeros(self.normalized_shape, device=self.device)
            
            if self.distribution == "boundary":
                weight = weight + torch.randn_like(weight) * 0.1
                bias = bias + torch.randn_like(bias) * 0.1
        else:
            weight = None
            bias = None

        meta = dict(
            distribution=self.distribution,
            normalized_shape=self.normalized_shape,
            eps=self.eps,
            elementwise_affine=self.elementwise_affine
        )
        return x, weight, bias, meta

