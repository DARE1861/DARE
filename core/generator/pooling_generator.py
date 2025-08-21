import torch
from typing import Dict, Any, Tuple

class PoolingInputGenerator:

    def __init__(
        self,
        input_shape: Tuple[int,int,int,int]=(1,3,224,224),
        pool_type: str = "max",
        kernel_size: int = 2,
        stride: int = 2,
        distribution: str = "normal",
        device: str = "cpu",
        seed: int = 2025
    ):
        self.input_shape = input_shape
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.distribution = distribution
        self.device = torch.device(device)
        torch.manual_seed(seed)

    def generate(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.distribution == "normal":
            x = torch.randn(self.input_shape, device=self.device)
        elif self.distribution == "uniform":
            x = (torch.rand(self.input_shape, device=self.device) - 0.5) * 2.0
        elif self.distribution == "boundary":
            tiny = torch.tensor(6.1e-5, device=self.device)
            sign = (torch.randint(0,2,self.input_shape,device=self.device)*2-1).float()
            x = (torch.rand(self.input_shape,device=self.device)*tiny)*sign + tiny*0.5
        else:
            raise ValueError(f"Unknown distribution {self.distribution}")

        meta = dict(
            distribution=self.distribution,
            pool_params=dict(pool_type=self.pool_type, kernel_size=self.kernel_size, stride=self.stride)
        )
        return x, meta
