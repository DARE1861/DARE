# precision_estimation/core/oracle/matmul_oracle_mc.py
"""
DataAwareMCMatmulOracle

Data-aware Monte Carlo Matmul error oracle (complete version)
- Supports multi-GPU parallelism, each worker creates corresponding Generator on its own device
- Supports element-level ULP noise simulation (input/weight/accumulation/output)
- Provides component error estimation (input/storage/accum/demote)
- Launches subprocesses via mp.get_context("spawn"), compatible with Linux
- Subprocess exceptions are returned through queue and handled uniformly by main process
"""

import os
import math
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import torch
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


class DataAwareMCMatmulOracle:
    """
    Data-aware Monte Carlo matrix multiplication error oracle
    Noise sources:
      - Input storage (A)
      - Weight storage (B)
      - Accumulation rounding (accum)
      - Output demotion (demote)
    """

    def __init__(
        self,
        strategy: PrecisionStrategy,
        num_mc_samples: int = 512,
        quantile: float = 0.999,
        safety_factor: float = 1.10,
        seeded: bool = True,
        devices: Optional[List[int]] = None,
        enable_noise_input: bool = True,
        enable_noise_weight: bool = True,
        enable_noise_accum: bool = True,
        enable_noise_output: bool = True,
    ):
        self.strategy = strategy
        self.num_mc_samples = int(num_mc_samples)
        self.quantile = float(quantile)
        self.safety_factor = float(safety_factor)
        self.seeded = bool(seeded)

        if devices is None:
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        self.devices = devices

        self.enable_noise_input = enable_noise_input
        self.enable_noise_weight = enable_noise_weight
        self.enable_noise_accum = enable_noise_accum
        self.enable_noise_output = enable_noise_output

    # ----- Reference output (FP64 -> FP32) -----
    @staticmethod
    @torch.inference_mode()
    def _compute_reference_on_device(A_cpu: torch.Tensor, B_cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Compute high-precision reference output on specified device"""
        A64 = A_cpu.to(device=device, dtype=torch.float64)
        B64 = B_cpu.to(device=device, dtype=torch.float64)
        Y64 = torch.matmul(A64, B64)
        return Y64.to(dtype=torch.float32)

    # ----- Worker main function -----
    @staticmethod
    def _worker_run(
        rank: int,
        device_id: Optional[int],
        A_cpu: torch.Tensor,
        B_cpu: torch.Tensor,
        strategy: PrecisionStrategy,
        num_local: int,
        noise_mask: Tuple[bool, bool, bool, bool],
        seed_base: int,
        return_queue: mp.Queue,
    ):
        """Worker function for parallel Monte Carlo sampling"""
        try:
            torch.set_num_threads(max(1, os.cpu_count() // 8))
            use_cuda = (device_id is not None) and torch.cuda.is_available()
            device = torch.device(f"cuda:{device_id}") if use_cuda else torch.device("cpu")
            if use_cuda:
                try:
                    torch.cuda.set_device(device)
                except Exception:
                    pass

            # Reference output: compute FP64 reference on current device
            Y_ref = DataAwareMCMatmulOracle._compute_reference_on_device(A_cpu, B_cpu, device)

            base_seed = int(seed_base) + 1337 * (rank + 1)
            errors: List[float] = []

            for i in range(num_local):
                g = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
                g.manual_seed(base_seed + i)

                # Move data and convert to float32 (main path)
                A = A_cpu.to(device=device, dtype=torch.float32)
                B = B_cpu.to(device=device, dtype=torch.float32)

                # Storage quantization (A/B)
                A_q = A.to(dtype=strategy.input_dtype).to(dtype=torch.float32)
                B_q = B.to(dtype=strategy.weight_dtype).to(dtype=torch.float32)

                # Promote to compute dtype
                A_c = A_q.to(dtype=strategy.compute_dtype)
                B_c = B_q.to(dtype=strategy.compute_dtype)

                # Input noise (A)
                if noise_mask[0]:
                    ulp_A = ulp_like(A_c, strategy.compute_dtype).to(device=device)
                    r = torch.rand(A_c.shape, generator=g, device=device, dtype=A_c.dtype)
                    A_c = A_c + (r - 0.5) * ulp_A

                # Weight noise (B)
                if noise_mask[1]:
                    ulp_B = ulp_like(B_c, strategy.compute_dtype).to(device=device)
                    r = torch.rand(B_c.shape, generator=g, device=device, dtype=B_c.dtype)
                    B_c = B_c + (r - 0.5) * ulp_B

                # Computation
                Y_c = torch.matmul(A_c, B_c)

                # Accumulation noise (approximation)
                if noise_mask[2]:
                    ulp_Y = ulp_like(Y_c, strategy.compute_dtype).to(device=device)
                    r = torch.rand(Y_c.shape, generator=g, device=device, dtype=Y_c.dtype)
                    Y_c = Y_c + (r - 0.5) * ulp_Y

                # Output demotion + output noise
                Y_out = Y_c.to(dtype=strategy.output_dtype).to(dtype=torch.float32)
                if noise_mask[3]:
                    ulp_O = ulp_like(Y_out, strategy.output_dtype).to(device=device)
                    r = torch.rand(Y_out.shape, generator=g, device=device, dtype=Y_out.dtype)
                    Y_out = Y_out + (r - 0.5) * ulp_O

                err = (Y_out - Y_ref).abs().max().item()
                errors.append(err)

            return_queue.put((rank, errors, None))

        except Exception as e:
            tb = traceback.format_exc()
            return_queue.put((rank, [], f"{repr(e)}\n{tb}"))

    # ----- Main function: predict error bound -----
    def predict_error_bound(self, A: torch.Tensor, B: torch.Tensor) -> OracleResult:
        """Predict error bound using Monte Carlo sampling"""
        A_cpu = A.detach().contiguous().cpu()
        B_cpu = B.detach().contiguous().cpu()

        noise_mask = (
            self.enable_noise_input,
            self.enable_noise_weight,
            self.enable_noise_accum,
            self.enable_noise_output,
        )

        # Single-machine CPU mode
        if len(self.devices) == 0:
            q = mp.Queue()
            self._worker_run(
                rank=0,
                device_id=None,
                A_cpu=A_cpu,
                B_cpu=B_cpu,
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
            # Multi-GPU
            per = math.ceil(self.num_mc_samples / max(1, len(self.devices)))
            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            procs = []
            for rank, dev in enumerate(self.devices):
                p = ctx.Process(
                    target=self._worker_run,
                    args=(
                        rank,
                        dev,
                        A_cpu,
                        B_cpu,
                        self.strategy,
                        per,
                        noise_mask,
                        1234 if self.seeded else int(time.time()),
                        q,
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

            all_errors = all_errors[: self.num_mc_samples]

        if len(all_errors) == 0:
            all_errors = [0.0]

        errs_tensor = torch.tensor(all_errors, dtype=torch.float32)
        qv = float(torch.quantile(errs_tensor, torch.tensor(self.quantile)))
        predicted = qv * self.safety_factor

        comp = self._estimate_components(A_cpu, B_cpu, num_samples=min(128, max(8, self.num_mc_samples // 8)))

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
                "op": "matmul",
            },
        )

    # ----- Error decomposition (median estimation with single-source switching) -----
    def _estimate_components(self, A_cpu: torch.Tensor, B_cpu: torch.Tensor, num_samples: int) -> Dict[str, float]:
        """Estimate component-wise error contributions"""
        def run(mask: Tuple[bool, bool, bool, bool]) -> float:
            device_id = self.devices[0] if len(self.devices) > 0 else None
            q = mp.Queue()
            self._worker_run(
                rank=0,
                device_id=device_id,
                A_cpu=A_cpu,
                B_cpu=B_cpu,
                strategy=self.strategy,
                num_local=num_samples,
                noise_mask=mask,
                seed_base=4321 if self.seeded else int(time.time()),
                return_queue=q,
            )
            _, errors, err_msg = q.get()
            if err_msg:
                raise RuntimeError(f"Component worker error: {err_msg}")
            if len(errors) == 0:
                return 0.0
            return float(torch.median(torch.tensor(errors, dtype=torch.float32)).item())

        return {
            "input_storage_error": run((True, False, False, False)),   # A
            "weight_storage_error": run((False, True, False, False)),  # B
            "accumulation_error": run((False, False, True, False)),
            "demote_error": run((False, False, False, True)),
        }