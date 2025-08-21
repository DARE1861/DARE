# precision_estimation/core/oracle/layernorm_oracle_mc.py
"""
DataAwareMCLayerNormOracle

Data-aware Monte Carlo LayerNorm error oracle
- Supports multi-GPU parallel processing
- Supports element-wise ULP noise simulation
- Analyzes two-stage errors: statistics computation error + normalization error
"""

import os
import math
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from core.config.precision_strategy import (
    PrecisionStrategy,
    ulp_like,
    quantize_to_dtype,
    promote_exact,
    demote_with_round,
)

@dataclass
class OracleResult:
    predicted_bound: float
    quantile: float
    safety_factor: float
    sample_errors: List[float]
    component_estimates: Dict[str, float]
    meta: Dict[str, Any]


class DataAwareMCLayerNormOracle:
    """
    Data-aware Monte Carlo LayerNorm error oracle
    
    LayerNorm computation flow:
    1. Compute mean: mean = x.mean(dim=normalized_dims)
    2. Compute variance: var = ((x - mean) ** 2).mean(dim=normalized_dims)
    3. Normalize: y = (x - mean) / sqrt(var + eps)
    4. Affine transform: output = y * weight + bias (if enabled)
    """

    def __init__(
        self,
        strategy: PrecisionStrategy,
        normalized_shape: Tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        num_mc_samples: int = 512,
        quantile: float = 0.999,
        safety_factor: float = 1.10,
        seeded: bool = True,
        devices: Optional[List[int]] = None,
        enable_noise_input: bool = True,
        enable_noise_weight: bool = True,
        enable_noise_bias: bool = True,
        enable_noise_stats: bool = True,
        enable_noise_output: bool = True,
    ):
        self.strategy = strategy
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.num_mc_samples = int(num_mc_samples)
        self.quantile = float(quantile)
        self.safety_factor = float(safety_factor)
        self.seeded = bool(seeded)

        if devices is None:
            if torch.cuda.is_available():
                devices = list(range(torch.cuda.device_count()))
            else:
                devices = []
        self.devices = devices

        # Noise switches
        self.enable_noise_input = enable_noise_input
        self.enable_noise_weight = enable_noise_weight
        self.enable_noise_bias = enable_noise_bias
        self.enable_noise_stats = enable_noise_stats  # Statistics computation noise
        self.enable_noise_output = enable_noise_output

    @staticmethod
    @torch.inference_mode()
    def _compute_reference_on_device(
        x_cpu: torch.Tensor, 
        weight_cpu: Optional[torch.Tensor], 
        bias_cpu: Optional[torch.Tensor], 
        device: torch.device, 
        normalized_shape: Tuple[int, ...],
        eps: float
    ) -> torch.Tensor:
        """
        Compute high-precision reference on given device
        """
        x64 = x_cpu.to(device=device, dtype=torch.float64)
        weight64 = weight_cpu.to(device=device, dtype=torch.float64) if weight_cpu is not None else None
        bias64 = bias_cpu.to(device=device, dtype=torch.float64) if bias_cpu is not None else None
        
        y64 = F.layer_norm(x64, normalized_shape, weight64, bias64, eps)
        return y64.to(dtype=torch.float32)

    @staticmethod
    def _layernorm_with_noise(
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        normalized_shape: Tuple[int, ...],
        eps: float,
        strategy: PrecisionStrategy,
        noise_mask: Tuple[bool, bool, bool, bool, bool],  # input, weight, bias, stats, output
        generator: torch.Generator,
        device: torch.device
    ) -> torch.Tensor:
        """
        LayerNorm implementation with noise, simulating various errors step by step
        """
        # Compute normalized dimensions
        input_shape = x.shape
        normalized_ndim = len(normalized_shape)
        normalized_dims = tuple(range(-normalized_ndim, 0))
        
        # Step 1: Input quantization and noise
        x_c = x.to(dtype=strategy.compute_dtype)
        if noise_mask[0]:  # input noise
            ulp_x = ulp_like(x_c, strategy.compute_dtype).to(device=device)
            r = torch.rand(x_c.shape, generator=generator, device=device, dtype=x_c.dtype)
            x_c = x_c + (r - 0.5) * ulp_x

        # Step 2: Weight/bias quantization and noise
        weight_c = None
        bias_c = None
        if weight is not None:
            weight_c = weight.to(dtype=strategy.compute_dtype)
            if noise_mask[1]:  # weight noise
                ulp_w = ulp_like(weight_c, strategy.compute_dtype).to(device=device)
                r = torch.rand(weight_c.shape, generator=generator, device=device, dtype=weight_c.dtype)
                weight_c = weight_c + (r - 0.5) * ulp_w
        
        if bias is not None:
            bias_c = bias.to(dtype=strategy.compute_dtype)
            if noise_mask[2]:  # bias noise
                ulp_b = ulp_like(bias_c, strategy.compute_dtype).to(device=device)
                r = torch.rand(bias_c.shape, generator=generator, device=device, dtype=bias_c.dtype)
                bias_c = bias_c + (r - 0.5) * ulp_b

        # Step 3: Statistics computation (mean and variance)
        mean = x_c.mean(dim=normalized_dims, keepdim=True)
        var = ((x_c - mean) ** 2).mean(dim=normalized_dims, keepdim=True)
        
        # Statistics noise
        if noise_mask[3]:  # stats noise
            ulp_mean = ulp_like(mean, strategy.compute_dtype).to(device=device)
            ulp_var = ulp_like(var, strategy.compute_dtype).to(device=device)
            
            r_mean = torch.rand(mean.shape, generator=generator, device=device, dtype=mean.dtype)
            r_var = torch.rand(var.shape, generator=generator, device=device, dtype=var.dtype)
            
            mean = mean + (r_mean - 0.5) * ulp_mean
            var = var + (r_var - 0.5) * ulp_var

        # Step 4: Normalization
        std = torch.sqrt(var + eps)
        y_c = (x_c - mean) / std

        # Step 5: Affine transformation
        if weight_c is not None:
            y_c = y_c * weight_c
        if bias_c is not None:
            y_c = y_c + bias_c

        # Step 6: Output demotion and noise
        y_out = y_c.to(dtype=strategy.output_dtype).to(dtype=torch.float32)
        if noise_mask[4]:  # output noise
            ulp_o = ulp_like(y_out, strategy.output_dtype).to(device=device)
            r = torch.rand(y_out.shape, generator=generator, device=device, dtype=y_out.dtype)
            y_out = y_out + (r - 0.5) * ulp_o

        return y_out

    @staticmethod
    def _worker_run(
        rank: int,
        device_id: Optional[int],
        x_cpu: torch.Tensor,
        weight_cpu: Optional[torch.Tensor],
        bias_cpu: Optional[torch.Tensor],
        normalized_shape: Tuple[int, ...],
        eps: float,
        strategy: PrecisionStrategy,
        num_local: int,
        noise_mask: Tuple[bool, bool, bool, bool, bool],
        seed_base: int,
        return_queue: mp.Queue,
    ):
        """
        Subprocess worker
        """
        try:
            start_worker = time.perf_counter()
            torch.set_num_threads(max(1, os.cpu_count() // 8))

            use_cuda = (device_id is not None) and torch.cuda.is_available()
            device = torch.device(f"cuda:{device_id}") if use_cuda else torch.device("cpu")
            if use_cuda:
                try:
                    torch.cuda.set_device(device)
                except Exception:
                    pass

            print(f"[worker {rank}] start. device_id={device_id}, use_cuda={use_cuda}, device={device}")
            
            # Compute reference value
            y_ref = DataAwareMCLayerNormOracle._compute_reference_on_device(
                x_cpu, weight_cpu, bias_cpu, device, normalized_shape, eps
            )

            base_seed = int(seed_base) + 1337 * (rank + 1)
            errors: List[float] = []

            for i in range(num_local):
                if device.type == 'cuda':
                    g = torch.Generator(device=device)
                else:
                    g = torch.Generator()
                g.manual_seed(base_seed + i)

                # Move data to device
                x = x_cpu.to(device=device, dtype=torch.float32)
                weight = weight_cpu.to(device=device, dtype=torch.float32) if weight_cpu is not None else None
                bias = bias_cpu.to(device=device, dtype=torch.float32) if bias_cpu is not None else None

                # Storage precision quantization
                x_q = x.to(dtype=strategy.input_dtype).to(dtype=torch.float32)
                weight_q = weight.to(dtype=strategy.weight_dtype).to(dtype=torch.float32) if weight is not None else None
                bias_q = bias.to(dtype=strategy.weight_dtype).to(dtype=torch.float32) if bias is not None else None  # bias uses weight precision

                # LayerNorm computation with noise
                y_out = DataAwareMCLayerNormOracle._layernorm_with_noise(
                    x_q, weight_q, bias_q, normalized_shape, eps, strategy, noise_mask, g, device
                )

                # Compute error
                err = (y_out - y_ref).abs().max().item()
                errors.append(err)

            end_worker = time.perf_counter()
            print(f"[worker {rank}] finished: total_worker_time={(end_worker-start_worker):.4f}s, generated {len(errors)} errors")
            
            return_queue.put((rank, errors, None))

        except Exception as e:
            tb = traceback.format_exc()
            return_queue.put((rank, [], f"{repr(e)}\n{tb}"))

    def predict_error_bound(
        self, 
        x: torch.Tensor, 
        weight: Optional[torch.Tensor] = None, 
        bias: Optional[torch.Tensor] = None
    ) -> OracleResult:
        """
        Predict LayerNorm error bound
        """
        x_cpu = x.detach().contiguous().cpu()
        weight_cpu = weight.detach().contiguous().cpu() if weight is not None else None
        bias_cpu = bias.detach().contiguous().cpu() if bias is not None else None

        noise_mask = (
            self.enable_noise_input,
            self.enable_noise_weight,
            self.enable_noise_bias,
            self.enable_noise_stats,
            self.enable_noise_output,
        )

        if len(self.devices) == 0:
            q = mp.Queue()
            self._worker_run(
                rank=0,
                device_id=None,
                x_cpu=x_cpu,
                weight_cpu=weight_cpu,
                bias_cpu=bias_cpu,
                normalized_shape=self.normalized_shape,
                eps=self.eps,
                strategy=self.strategy,
                num_local=self.num_mc_samples,
                noise_mask=noise_mask,
                seed_base=1234 if self.seeded else int(time.time()),
                return_queue=q,
            )
            _, errors, err_msg = q.get()
            if err_msg:
                raise RuntimeError(f"Worker error: {err_msg}")
            all_errors = errors
        else:
            per = math.ceil(self.num_mc_samples / max(1, len(self.devices)))
            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            procs = []
            for rank, dev in enumerate(self.devices):
                p = ctx.Process(
                    target=self._worker_run,
                    args=(
                        rank, dev, x_cpu, weight_cpu, bias_cpu, 
                        self.normalized_shape, self.eps, self.strategy, 
                        per, noise_mask, 1234 if self.seeded else int(time.time()), q,
                    ),
                )
                p.daemon = True
                p.start()
                procs.append(p)

            all_errors: List[float] = []
            any_error: Optional[str] = None

            for _ in range(len(procs)):
                _, errors, err_msg = q.get()
                if err_msg and any_error is None:
                    any_error = err_msg
                all_errors.extend(errors)

            for p in procs:
                p.join(timeout=60)
            for p in procs:
                if p.is_alive():
                    p.terminate()

            if any_error:
                raise RuntimeError(f"Worker error: {any_error}")

            all_errors = all_errors[:self.num_mc_samples]

        if len(all_errors) == 0:
            all_errors = [0.0]

        errs_tensor = torch.tensor(all_errors, dtype=torch.float32)
        qv = float(torch.quantile(errs_tensor, torch.tensor(self.quantile)))
        predicted = qv * self.safety_factor

        comp = self._estimate_components(x_cpu, weight_cpu, bias_cpu, num_samples=min(128, max(8, self.num_mc_samples // 8)))

        return OracleResult(
            predicted_bound=predicted,
            quantile=self.quantile,
            safety_factor=self.safety_factor,
            sample_errors=all_errors,
            component_estimates=comp,
            meta={
                "num_samples": len(all_errors),
                "devices": self.devices,
                "strategy": str(self.strategy),
                "noise_mask": noise_mask,
                "normalized_shape": self.normalized_shape,
                "eps": self.eps,
                "elementwise_affine": self.elementwise_affine,
            },
        )

    def _estimate_components(
        self, 
        x_cpu: torch.Tensor, 
        weight_cpu: Optional[torch.Tensor], 
        bias_cpu: Optional[torch.Tensor], 
        num_samples: int
    ) -> Dict[str, float]:
        """
        Estimate error components
        """
        def run(mask: Tuple[bool, bool, bool, bool, bool]) -> float:
            device_id = self.devices[0] if len(self.devices) > 0 else None
            q = mp.Queue()
            self._worker_run(
                rank=0, device_id=device_id, x_cpu=x_cpu, weight_cpu=weight_cpu, bias_cpu=bias_cpu,
                normalized_shape=self.normalized_shape, eps=self.eps, strategy=self.strategy,
                num_local=num_samples, noise_mask=mask,
                seed_base=4321 if self.seeded else int(time.time()), return_queue=q,
            )
            _, errors, err_msg = q.get()
            if err_msg:
                raise RuntimeError(f"Component worker error: {err_msg}")
            if len(errors) == 0:
                return 0.0
            return float(torch.median(torch.tensor(errors, dtype=torch.float32)).item())

        components = {
            "input_storage_error": run((True, False, False, False, False)),
            "statistics_computation_error": run((False, False, False, True, False)),
            "output_storage_error": run((False, False, False, False, True)),
        }
        
        # Only estimate corresponding errors when learnable parameters exist
        if weight_cpu is not None:
            components["weight_storage_error"] = run((False, True, False, False, False))
        if bias_cpu is not None:
            components["bias_storage_error"] = run((False, False, True, False, False))

        return components