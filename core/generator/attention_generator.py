# precision_estimation/core/generator/attention_generator.py
import torch
import math
from typing import Dict, Any, Tuple

class AttentionInputGenerator:


    def __init__(
        self,
        batch_size: int = 1,
        seq_len: int = 512,
        d_model: int = 768,
        num_heads: int = 12,
        distribution: str = "normal",
        device: str = "cpu",
        seed: int = 2025
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.distribution = distribution
        self.device = torch.device(device)
        torch.manual_seed(seed)

    def generate(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Returns: q, k, v, mask, meta
        """
        q_shape = (self.batch_size, self.num_heads, self.seq_len, self.d_head)
        k_shape = (self.batch_size, self.num_heads, self.seq_len, self.d_head)
        v_shape = (self.batch_size, self.num_heads, self.seq_len, self.d_head)
        mask_shape = (self.batch_size, self.num_heads, self.seq_len, self.seq_len)

        if self.distribution == "normal":
            q = torch.randn(q_shape, device=self.device) * 0.1
            k = torch.randn(k_shape, device=self.device) * 0.1
            v = torch.randn(v_shape, device=self.device) * 0.1
        elif self.distribution == "uniform":
            q = (torch.rand(q_shape, device=self.device) - 0.5) * 0.2
            k = (torch.rand(k_shape, device=self.device) - 0.5) * 0.2
            v = (torch.rand(v_shape, device=self.device) - 0.5) * 0.2
        elif self.distribution == "boundary":

            tiny = torch.tensor(1e-4, device=self.device)
            q = torch.randn(q_shape, device=self.device) * tiny
            k = torch.randn(k_shape, device=self.device) * tiny
            v = torch.randn(v_shape, device=self.device) * 0.1
        elif self.distribution == "adversarial_sum":

            base_scale = 1e-2
            q = torch.randn(q_shape, device=self.device) * base_scale
            k = torch.randn(k_shape, device=self.device) * base_scale
            v = torch.randn(v_shape, device=self.device) * base_scale

            noise = (torch.rand_like(q) - 0.5) * 1e-7
            q = q + noise
            k = k + noise[:, :, :k.shape[2], :]
            v = v + noise[:, :, :v.shape[2], :]
        elif self.distribution == "attention_specific":
            scale = 0.05
            base_pattern = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.d_head, device=self.device) * scale
            q = base_pattern + torch.randn_like(base_pattern) * scale * 0.1
            k = base_pattern + torch.randn_like(base_pattern) * scale * 0.1
            v = torch.randn(v_shape, device=self.device) * scale
        else:
            raise ValueError(f"Unknown distribution {self.distribution}")

        mask = torch.tril(torch.ones(mask_shape, device=self.device)).bool()
        
        attention_params = dict(
            scale=1.0 / math.sqrt(self.d_head),
            is_causal=True
        )
        meta = dict(
            distribution=self.distribution,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_head=self.d_head,
            attention_params=attention_params
        )
        return q, k, v, mask, meta

