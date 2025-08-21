# precision_estimation/core/oracle/batchnorm_oracle_mc.py
"""
DataAwareMCBatchNormOracle

Data-aware Monte Carlo BatchNorm error oracle
- Supports multi-GPU parallel processing
- Supports element-wise ULP noise simulation (input/weight/statistics computation/output)
- Provides component error estimation (input/mean/var/affine/demote)
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


class DataAwareMCBatchNormOracle:
    """
    Data-aware Monte Carlo BatchNorm error oracle
    """

    def __init__(
        self,
        strategy: PrecisionStrategy,
        bn_params: Dict[str, Any],
        num_mc_samples: int = 512,
        quantile: float = 0.999,
        safety_factor: float = 1.10,
        seeded: bool = True,
        devices: Optional[List[int]] = None,
        enable_noise_input: bool = True,
        enable_noise_weight: bool = True,
        enable_noise_stats: bool = True,
        enable_noise_output: bool = True,
    ):
        self.strategy = strategy
        self.params = {
            "eps": bn_params.get("eps", 1e-5),
            "momentum": bn_params.get("momentum", 0.1),
            "affine": bn_params.get("affine", True),
            "track_running_stats": bn_params.get("track_running_stats", True),
            "training": bn_params.get("training", True),
        }
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

        self.enable_noise_input = enable_noise_input
        self.enable_noise_weight = enable_noise_weight
        self.enable_noise_stats = enable_noise_stats
        self.enable_noise_output = enable_noise_output

    @staticmethod
    @torch.inference_mode()
    def _compute_reference_on_device(
        x_cpu: torch.Tensor, 
        weight_cpu: Optional[torch.Tensor], 
        bias_cpu: Optional[torch.Tensor],
        running_mean_cpu: torch.Tensor,
        running_var_cpu: torch.Tensor,
        device: torch.device, 
        params: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute high-precision reference on given device
        """
        x64 = x_cpu.to(device=device, dtype=torch.float64)
        weight64 = weight_cpu.to(device=device, dtype=torch.float64) if weight_cpu is not None else None
        bias64 = bias_cpu.to(device=device, dtype=torch.float64) if bias_cpu is not None else None
        running_mean64 = running_mean_cpu.to(device=device, dtype=torch.float64)
        running_var64 = running_var_cpu.to(device=device, dtype=torch.float64)
        
        y64 = F.batch_norm(
            x64, running_mean64, running_var64, weight64, bias64,
            training=params["training"],
            momentum=params["momentum"],
            eps=params["eps"]
        )
        return y64.to(dtype=torch.float32)

    @staticmethod
    def _worker_run(
        rank: int,
        device_id: Optional[int],
        x_cpu: torch.Tensor,
        weight_cpu: Optional[torch.Tensor],
        bias_cpu: Optional[torch.Tensor],
        running_mean_cpu: torch.Tensor,
        running_var_cpu: torch.Tensor,
        params: Dict[str, Any],
        strategy: PrecisionStrategy,
        num_local: int,
        noise_mask: Tuple[bool, bool, bool, bool],
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
            y_ref = DataAwareMCBatchNormOracle._compute_reference_on_device(
                x_cpu, weight_cpu, bias_cpu, running_mean_cpu, running_var_cpu, device, params
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
                running_mean = running_mean_cpu.to(device=device, dtype=torch.float32)
                running_var = running_var_cpu.to(device=device, dtype=torch.float32)

                # Storage precision quantization
                x_q = x.to(dtype=strategy.input_dtype).to(dtype=torch.float32)
                if weight is not None:
                    weight_q = weight.to(dtype=strategy.weight_dtype).to(dtype=torch.float32)
                    bias_q = bias.to(dtype=strategy.weight_dtype).to(dtype=torch.float32) if bias is not None else None
                else:
                    weight_q = None
                    bias_q = None

                # Promote to compute precision
                x_c = x_q.to(dtype=strategy.compute_dtype)
                weight_c = weight_q.to(dtype=strategy.compute_dtype) if weight_q is not None else None
                bias_c = bias_q.to(dtype=strategy.compute_dtype) if bias_q is not None else None
                running_mean_c = running_mean.to(dtype=strategy.compute_dtype)
                running_var_c = running_var.to(dtype=strategy.compute_dtype)

                # Input noise
                if noise_mask[0]:
                    ulp_x = ulp_like(x_c, strategy.compute_dtype).to(device=device)
                    r = torch.rand(x_c.shape, generator=g, device=device, dtype=x_c.dtype)
                    x_c = x_c + (r - 0.5) * ulp_x

                # Weight/bias noise
                if noise_mask[1] and weight_c is not None:
                    ulp_w = ulp_like(weight_c, strategy.compute_dtype).to(device=device)
                    r = torch.rand(weight_c.shape, generator=g, device=device, dtype=weight_c.dtype)
                    weight_c = weight_c + (r - 0.5) * ulp_w
                    
                    if bias_c is not None:
                        ulp_b = ulp_like(bias_c, strategy.compute_dtype).to(device=device)
                        r = torch.rand(bias_c.shape, generator=g, device=device, dtype=bias_c.dtype)
                        bias_c = bias_c + (r - 0.5) * ulp_b

                # Statistics computation noise (simulate accumulation error in mean/variance computation)
                if noise_mask[2]:
                    if params["training"]:
                        # In training mode, mean and variance are computed from current batch
                        # We simulate accumulation error in this computation process
                        N = x_c.shape[0]
                        if N > 1:
                            # Simulate mean computation error
                            mean_err_scale = math.sqrt(N) * 0.5  # Heuristic estimate of accumulation error
                            mean_ulp = ulp_like(x_c.mean(dim=(0, 2, 3), keepdim=True), strategy.compute_dtype)
                            # Use torch.randn instead of randn_like to support generator parameter
                            mean_noise = torch.randn(mean_ulp.shape, generator=g, device=device, dtype=mean_ulp.dtype) * mean_ulp * mean_err_scale
                            
                            # Simulate variance computation error
                            var_ulp = ulp_like(x_c.var(dim=(0, 2, 3), keepdim=True, unbiased=False), strategy.compute_dtype)
                            var_noise = torch.randn(var_ulp.shape, generator=g, device=device, dtype=var_ulp.dtype) * var_ulp * mean_err_scale
                            
                            # Directly inject noise to running statistics before BatchNorm computation
                            running_mean_c = running_mean_c + mean_noise.squeeze()
                            running_var_c = running_var_c + var_noise.squeeze()

                # Execute BatchNorm
                y_c = F.batch_norm(
                    x_c, running_mean_c, running_var_c, weight_c, bias_c,
                    training=params["training"],
                    momentum=params["momentum"],
                    eps=params["eps"]
                )

                # Output demotion and simulate output storage error
                y_out = y_c.to(dtype=strategy.output_dtype).to(dtype=torch.float32)
                if noise_mask[3]:
                    ulp_o = ulp_like(y_out, strategy.output_dtype).to(device=device)
                    r = torch.rand(y_out.shape, generator=g, device=device, dtype=y_out.dtype)
                    y_out = y_out + (r - 0.5) * ulp_o

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
        weight: Optional[torch.Tensor], 
        bias: Optional[torch.Tensor],
        running_mean: torch.Tensor,
        running_var: torch.Tensor
    ) -> OracleResult:
        """
        Predict BatchNorm error bound
        """
        x_cpu = x.detach().contiguous().cpu()
        weight_cpu = weight.detach().contiguous().cpu() if weight is not None else None
        bias_cpu = bias.detach().contiguous().cpu() if bias is not None else None
        running_mean_cpu = running_mean.detach().contiguous().cpu()
        running_var_cpu = running_var.detach().contiguous().cpu()

        noise_mask = (
            self.enable_noise_input,
            self.enable_noise_weight,
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
                running_mean_cpu=running_mean_cpu,
                running_var_cpu=running_var_cpu,
                params=self.params,
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
                        running_mean_cpu, running_var_cpu, self.params, self.strategy, 
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

        comp = self._estimate_components(
            x_cpu, weight_cpu, bias_cpu, running_mean_cpu, running_var_cpu,
            num_samples=min(128, max(8, self.num_mc_samples // 8))
        )

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
            },
        )

    def _estimate_components(
        self, 
        x_cpu: torch.Tensor, 
        weight_cpu: Optional[torch.Tensor], 
        bias_cpu: Optional[torch.Tensor],
        running_mean_cpu: torch.Tensor,
        running_var_cpu: torch.Tensor,
        num_samples: int
    ) -> Dict[str, float]:
        """
        Estimate error components
        """
        def run(mask: Tuple[bool, bool, bool, bool]) -> float:
            device_id = self.devices[0] if len(self.devices) > 0 else None
            q = mp.Queue()
            self._worker_run(
                rank=0, device_id=device_id, 
                x_cpu=x_cpu, weight_cpu=weight_cpu, bias_cpu=bias_cpu,
                running_mean_cpu=running_mean_cpu, running_var_cpu=running_var_cpu,
                params=self.params, strategy=self.strategy,
                num_local=num_samples, noise_mask=mask,
                seed_base=4321 if self.seeded else int(time.time()), return_queue=q,
            )
            _, errors, err_msg = q.get()
            if err_msg:
                raise RuntimeError(f"Component worker error: {err_msg}")
            if len(errors) == 0:
                return 0.0
            return float(torch.median(torch.tensor(errors, dtype=torch.float32)).item())

        return {
            "input_storage_error": run((True, False, False, False)),
            "affine_params_error": run((False, True, False, False)),
            "statistics_computation_error": run((False, False, True, False)),
            "output_storage_error": run((False, False, False, True)),
        }