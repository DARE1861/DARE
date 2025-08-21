# precision_estimation/core/oracle/softmax_oracle_mc.py
"""
DataAwareMCSoftmaxOracle

Data-aware Monte Carlo Softmax error oracle
- Supports multi-GPU parallel processing
- Supports element-wise ULP noise simulation
- Specially handles Softmax numerical stability issues
- Provides component error estimation (exp/max_sub/sum/norm)
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


class DataAwareMCSoftmaxOracle:
    """
    Data-aware Monte Carlo Softmax error oracle
    
    Special consideration for Softmax numerical stability:
    - Error propagation from max subtraction trick
    - Precision loss in exponential operations
    - Sum accumulation errors
    - Normalization division errors
    """

    def __init__(
        self,
        strategy: PrecisionStrategy,
        dim: int = -1,
        num_mc_samples: int = 512,
        quantile: float = 0.999,
        safety_factor: float = 1.10,
        seeded: bool = True,
        devices: Optional[List[int]] = None,
        enable_noise_input: bool = True,
        enable_noise_max_sub: bool = True,
        enable_noise_exp: bool = True,
        enable_noise_sum: bool = True,
        enable_noise_output: bool = True,
    ):
        self.strategy = strategy
        self.dim = dim
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

        # Softmax-specific noise switches
        self.enable_noise_input = enable_noise_input
        self.enable_noise_max_sub = enable_noise_max_sub
        self.enable_noise_exp = enable_noise_exp
        self.enable_noise_sum = enable_noise_sum
        self.enable_noise_output = enable_noise_output

    @staticmethod
    @torch.inference_mode()
    def _compute_reference_on_device(x_cpu: torch.Tensor, device: torch.device, dim: int) -> torch.Tensor:
        """
        Compute high-precision reference on given device
        """
        x64 = x_cpu.to(device=device, dtype=torch.float64)
        y64 = F.softmax(x64, dim=dim)
        return y64.to(dtype=torch.float32)

    @staticmethod
    def _softmax_with_noise(
        x: torch.Tensor,
        dim: int,
        strategy: PrecisionStrategy,
        noise_mask: Tuple[bool, bool, bool, bool, bool],
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Execute noisy softmax, simulating errors at each computational stage step by step
        """
        device = x.device
        dtype = strategy.compute_dtype

        # 1. Input noise
        if noise_mask[0]:  # input noise
            ulp_x = ulp_like(x, dtype).to(device=device)
            r = torch.rand(x.shape, generator=generator, device=device, dtype=dtype)
            x = x + (r - 0.5) * ulp_x

        # 2. Max subtraction for numerical stability
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
        
        # 2a. Max subtraction noise
        if noise_mask[1]:  # max_sub noise
            ulp_max = ulp_like(x_max, dtype).to(device=device)
            r = torch.rand(x_max.shape, generator=generator, device=device, dtype=dtype)
            x_max = x_max + (r - 0.5) * ulp_max
        
        x_shifted = x - x_max

        # 3. Exponential computation
        x_exp = torch.exp(x_shifted)
        
        # 3a. Exponential noise
        if noise_mask[2]:  # exp noise
            ulp_exp = ulp_like(x_exp, dtype).to(device=device)
            r = torch.rand(x_exp.shape, generator=generator, device=device, dtype=dtype)
            x_exp = x_exp + (r - 0.5) * ulp_exp

        # 4. Sum computation
        exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
        
        # 4a. Sum accumulation noise
        if noise_mask[3]:  # sum noise
            ulp_sum = ulp_like(exp_sum, dtype).to(device=device)
            r = torch.rand(exp_sum.shape, generator=generator, device=device, dtype=dtype)
            exp_sum = exp_sum + (r - 0.5) * ulp_sum

        # 5. Division (normalization)
        y = x_exp / exp_sum.clamp(min=1e-20)  # Prevent division by zero

        # 5a. Output precision noise
        if noise_mask[4]:  # output noise
            ulp_out = ulp_like(y, strategy.output_dtype).to(device=device)
            r = torch.rand(y.shape, generator=generator, device=device, dtype=y.dtype)
            y = y + (r - 0.5) * ulp_out

        return y

    @staticmethod
    def _worker_run(
        rank: int,
        device_id: Optional[int],
        x_cpu: torch.Tensor,
        dim: int,
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
            y_ref = DataAwareMCSoftmaxOracle._compute_reference_on_device(x_cpu, device, dim)

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

                # Storage precision quantization
                x_q = x.to(dtype=strategy.input_dtype).to(dtype=torch.float32)

                # Promote to compute precision
                x_c = x_q.to(dtype=strategy.compute_dtype)

                # Execute noisy softmax
                y_noisy = DataAwareMCSoftmaxOracle._softmax_with_noise(
                    x_c, dim, strategy, noise_mask, g
                )

                # Output demotion
                y_out = y_noisy.to(dtype=strategy.output_dtype).to(dtype=torch.float32)

                # Compute error
                err = (y_out - y_ref).abs().max().item()
                errors.append(err)

            end_worker = time.perf_counter()
            print(f"[worker {rank}] finished: total_worker_time={(end_worker-start_worker):.4f}s, generated {len(errors)} errors")
            
            return_queue.put((rank, errors, None))

        except Exception as e:
            tb = traceback.format_exc()
            return_queue.put((rank, [], f"{repr(e)}\n{tb}"))

    def predict_error_bound(self, x: torch.Tensor) -> OracleResult:
        """
        Predict Softmax error bound
        """
        x_cpu = x.detach().contiguous().cpu()
        noise_mask = (
            self.enable_noise_input,
            self.enable_noise_max_sub,
            self.enable_noise_exp,
            self.enable_noise_sum,
            self.enable_noise_output,
        )

        if len(self.devices) == 0:
            q = mp.Queue()
            self._worker_run(
                rank=0,
                device_id=None,
                x_cpu=x_cpu,
                dim=self.dim,
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
                        rank, dev, x_cpu, self.dim, self.strategy, per, noise_mask,
                        1234 if self.seeded else int(time.time()), q,
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

        comp = self._estimate_components(x_cpu, num_samples=min(128, max(8, self.num_mc_samples // 8)))

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
                "dim": self.dim,
            },
        )

    def _estimate_components(self, x_cpu: torch.Tensor, num_samples: int) -> Dict[str, float]:
        """
        Estimate error components for each stage of Softmax
        """
        def run(mask: Tuple[bool, bool, bool, bool, bool]) -> float:
            device_id = self.devices[0] if len(self.devices) > 0 else None
            q = mp.Queue()
            self._worker_run(
                rank=0, device_id=device_id, x_cpu=x_cpu, dim=self.dim,
                strategy=self.strategy, num_local=num_samples, noise_mask=mask,
                seed_base=4321 if self.seeded else int(time.time()), return_queue=q,
            )
            _, errors, err_msg = q.get()
            if err_msg:
                raise RuntimeError(f"Component worker error: {err_msg}")
            if len(errors) == 0:
                return 0.0
            return float(torch.median(torch.tensor(errors, dtype=torch.float32)).item())

        return {
            "input_storage_error": run((True, False, False, False, False)),
            "max_subtraction_error": run((False, True, False, False, False)),
            "exponential_error": run((False, False, True, False, False)),
            "sum_accumulation_error": run((False, False, False, True, False)),
            "output_storage_error": run((False, False, False, False, True)),
        }