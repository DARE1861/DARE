# precision_estimation/core/oracle/conv2d_oracle_mc.py
"""
DataAwareMCConv2DOracle

Data-aware Monte Carlo convolution error oracle (complete version)
- Supports multi-GPU parallel processing, each worker creates corresponding Generator on its own device
- Supports element-wise ULP noise simulation (input/weight/accumulation/output)
- Provides component error estimates (input/storage/accum/demote)
- Launches subprocesses via mp.get_context("spawn"), compatible with Linux
- Subprocess exceptions returned via queue, handled uniformly by main process
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

# Result data class
@dataclass
class OracleResult:
    predicted_bound: float
    quantile: float
    safety_factor: float
    sample_errors: List[float]
    component_estimates: Dict[str, float]
    meta: Dict[str, Any]


class DataAwareMCConv2DOracle:
    """
    Data-aware Monte Carlo convolution error oracle (complete implementation with detailed implementation details)

    Parameter explanation (constructor):
      - strategy: PrecisionStrategy object, defines input/weight/compute/output dtype
      - conv_params: dict, convolution parameters (stride,padding,dilation,groups)
      - num_mc_samples: total Monte Carlo sampling count
      - quantile: quantile point for prediction boundary (e.g. 0.999)
      - safety_factor: safety factor for amplifying quantile
      - seeded: whether to use fixed seed for reproducibility
      - devices: GPU id list; empty means run on CPU
      - enable_noise_*: corresponding noise switches
    """

    def __init__(
        self,
        strategy: PrecisionStrategy,
        conv_params: Dict[str, Any],
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
        self.params = {
            "stride": conv_params.get("stride", 1),
            "padding": conv_params.get("padding", 0),
            "dilation": conv_params.get("dilation", 1),
            "groups": conv_params.get("groups", 1),
        }
        self.num_mc_samples = int(num_mc_samples)
        self.quantile = float(quantile)
        self.safety_factor = float(safety_factor)
        self.seeded = bool(seeded)

        # Default devices: all visible cuda devices; otherwise CPU (devices = [])
        if devices is None:
            if torch.cuda.is_available():
                devices = list(range(torch.cuda.device_count()))
            else:
                devices = []
        self.devices = devices

        # Noise switches
        self.enable_noise_input = enable_noise_input
        self.enable_noise_weight = enable_noise_weight
        self.enable_noise_accum = enable_noise_accum
        self.enable_noise_output = enable_noise_output

    # ----- Helper: compute FP64 reference output (called within worker) -----
    @staticmethod
    @torch.inference_mode()
    def _compute_reference_on_device(x_cpu: torch.Tensor, w_cpu: torch.Tensor, device: torch.device, params: Dict[str, Any]) -> torch.Tensor:
        """
        On given device (usually CPU or GPU), convert x_cpu/w_cpu to float64 and perform convolution, return float32 result.
        Note: If device is GPU, we compute high-precision reference on GPU (may be slow), but ensures comparison consistency.
        """
        x64 = x_cpu.to(device=device, dtype=torch.float64)
        w64 = w_cpu.to(device=device, dtype=torch.float64)
        y64 = F.conv2d(x64, w64,
                       bias=None,
                       stride=params["stride"],
                       padding=params["padding"],
                       dilation=params["dilation"],
                       groups=params["groups"])
        return y64.to(dtype=torch.float32)

    # ----- Worker main body -----
    @staticmethod
    def _worker_run(
        rank: int,
        device_id: Optional[int],
        x_cpu: torch.Tensor,
        w_cpu: torch.Tensor,
        params: Dict[str, Any],
        strategy: PrecisionStrategy,
        num_local: int,
        noise_mask: Tuple[bool, bool, bool, bool],
        seed_base: int,
        return_queue: mp.Queue,
    ):
        """
        Subprocess worker:
        - rank: current worker index (used for seed construction)
        - device_id: specified GPU id or None (indicates CPU)
        - x_cpu/w_cpu: master copy on CPU (subprocess will move to device internally)
        - num_local: number of samples this worker needs to execute
        - noise_mask: (input, weight, accum, output) noise switches
        - seed_base: overall seed base (for reproducibility)
        - return_queue: put (rank, errors_list, err_msg_or_None) back to main process
        """
        try:
            import time
            start_worker = time.perf_counter()
            # Limit subprocess thread count to avoid oversubscription
            torch.set_num_threads(max(1, os.cpu_count() // 8))

            use_cuda = (device_id is not None) and torch.cuda.is_available()
            device = torch.device(f"cuda:{device_id}") if use_cuda else torch.device("cpu")
            if use_cuda:
                # Explicitly set GPU used by current process
                try:
                    torch.cuda.set_device(device)
                except Exception:
                    pass

            # --- DEBUG PRINT ---
            print(f"[worker {rank}] start. device_id={device_id}, use_cuda={use_cuda}, device={device}")
            print(f"[worker {rank}] torch.cuda.is_available()={torch.cuda.is_available()}, cuda_count={torch.cuda.device_count()}")
            if use_cuda:
                try:
                    print(f"[worker {rank}] device name: {torch.cuda.get_device_name(device_id)}")
                    print(f"[worker {rank}] memory_allocated(before) = {torch.cuda.memory_allocated(device):,}")
                except Exception as e:
                    print(f"[worker {rank}] get_device_name/error: {e}")
            # -------------------
            # First compute reference y_ref (compute fp64 reference on same device)
            y_ref = DataAwareMCConv2DOracle._compute_reference_on_device(x_cpu, w_cpu, device, params)

            # Random seed base (ensure each worker is different)
            base_seed = int(seed_base) + 1337 * (rank + 1)

            errors: List[float] = []

            # Each simulation: use Generator matching the device
            for i in range(num_local):
                # Create generator for this sampling
                if device.type == 'cuda':
                    # Create generator on CUDA device (avoid CPU generator with CUDA tensor mismatch)
                    g = torch.Generator(device=device)
                else:
                    g = torch.Generator()

                g.manual_seed(base_seed + i)

                # Move data to device and quantize according to storage precision
                x = x_cpu.to(device=device, dtype=torch.float32)
                w = w_cpu.to(device=device, dtype=torch.float32)

                # Storage precision quantization (simulate storage/load error)
                x_q = x.to(dtype=strategy.input_dtype).to(dtype=torch.float32)
                w_q = w.to(dtype=strategy.weight_dtype).to(dtype=torch.float32)

                # Promote to compute precision
                x_c = x_q.to(dtype=strategy.compute_dtype)
                w_c = w_q.to(dtype=strategy.compute_dtype)

                # ---- Input noise (based on element-wise ULP) ----
                if noise_mask[0]:
                    # ulp_like returns element-wise ULP estimate (sign/numerically stable)
                    ulp_x = ulp_like(x_c, strategy.compute_dtype).to(device=device)
                    # Generate [0,1) random with shape consistent with x_c
                    r = torch.rand(x_c.shape, generator=g, device=device, dtype=x_c.dtype)
                    x_c = x_c + (r - 0.5) * ulp_x

                # ---- Weight noise ----
                if noise_mask[1]:
                    ulp_w = ulp_like(w_c, strategy.compute_dtype).to(device=device)
                    r = torch.rand(w_c.shape, generator=g, device=device, dtype=w_c.dtype)
                    w_c = w_c + (r - 0.5) * ulp_w

                # ---- Execute convolution (compute stage) ----
                y_c = F.conv2d(
                    x_c, w_c, bias=None,
                    stride=params["stride"],
                    padding=params["padding"],
                    dilation=params["dilation"],
                    groups=params["groups"],
                )

                # ---- Accumulation noise (approximation) ----
                if noise_mask[2]:
                    ulp_y = ulp_like(y_c, strategy.compute_dtype).to(device=device)
                    r = torch.rand(y_c.shape, generator=g, device=device, dtype=y_c.dtype)
                    y_c = y_c + (r - 0.5) * ulp_y

                # ---- Output demotion and simulate output storage error ----
                y_out = y_c.to(dtype=strategy.output_dtype).to(dtype=torch.float32)
                if noise_mask[3]:
                    ulp_o = ulp_like(y_out, strategy.output_dtype).to(device=device)
                    r = torch.rand(y_out.shape, generator=g, device=device, dtype=y_out.dtype)
                    y_out = y_out + (r - 0.5) * ulp_o

                # Compute maximum error for current sample (compare with reference y_ref)
                err = (y_out - y_ref).abs().max().item()
                errors.append(err)

            # Output worker total time and memory before function return
            end_worker = time.perf_counter()
            if use_cuda:
                try:
                    print(f"[worker {rank}] memory_allocated(after) = {torch.cuda.memory_allocated(device):,}")
                    print(f"[worker {rank}] max_memory_allocated = {torch.cuda.max_memory_allocated(device):,}")
                except:
                    pass
            print(f"[worker {rank}] finished: total_worker_time={(end_worker-start_worker):.4f}s, generated {len(errors)} errors")
        
            # Normal completion, put result back to main process
            return_queue.put((rank, errors, None))

        except Exception as e:
            # Catch exception and send traceback back to main process (avoid main process waiting indefinitely)
            tb = traceback.format_exc()
            return_queue.put((rank, [], f"{repr(e)}\n{tb}"))

    # ----- Main call: predict error bound -----
    def predict_error_bound(self, x: torch.Tensor, w: torch.Tensor) -> OracleResult:
        """
        In main process, distribute subprocesses for Monte Carlo simulation and aggregate:
        - Move x,w back to CPU as master copy (for serialization)
        - Move to corresponding device in each worker and execute num_local simulations
        """
        # master copy on CPU (avoid serializing large tensors to GPU)
        x_cpu = x.detach().contiguous().cpu()
        w_cpu = w.detach().contiguous().cpu()

        noise_mask = (
            self.enable_noise_input,
            self.enable_noise_weight,
            self.enable_noise_accum,
            self.enable_noise_output,
        )

        # If no GPU (devices == []), directly call _worker_run in single CPU worker (avoid spawn overhead)
        if len(self.devices) == 0:
            q = mp.Queue()
            # Single process direct call, don't create new process
            self._worker_run(
                rank=0,
                device_id=None,
                x_cpu=x_cpu,
                w_cpu=w_cpu,
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
            # Multi-GPU: spawn multiple subprocesses, each process handles per samples
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
                        x_cpu,
                        w_cpu,
                        self.params,
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

            # Collect results from each subprocess
            for _ in range(len(procs)):
                _, errors, err_msg = q.get()
                if err_msg and any_error is None:
                    any_error = err_msg
                all_errors.extend(errors)

            # Wait for processes to finish and cleanup (timeout protection)
            for p in procs:
                p.join(timeout=60)
            for p in procs:
                if p.is_alive():
                    p.terminate()

            if any_error:
                raise RuntimeError(f"Worker error: {any_error}")

            # Ensure truncation to expected count
            all_errors = all_errors[: self.num_mc_samples]

        # If no samples (exceptional case), return conservative estimate of 0
        if len(all_errors) == 0:
            all_errors = [0.0]

        errs_tensor = torch.tensor(all_errors, dtype=torch.float32)
        qv = float(torch.quantile(errs_tensor, torch.tensor(self.quantile)))
        predicted = qv * self.safety_factor

        # Component error estimation (separate small sample experiment)
        comp = self._estimate_components(x_cpu, w_cpu, num_samples=min(128, max(8, self.num_mc_samples // 8)))

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

    # ----- Error decomposition: separately enable only one noise source to estimate its median impact -----
    def _estimate_components(self, x_cpu: torch.Tensor, w_cpu: torch.Tensor, num_samples: int) -> Dict[str, float]:
        """
        For 4 types of noise (input/weight/accum/demote), run separate short MC experiments to estimate typical size of each error type.
        Returns a dictionary with values as median errors (more robust).
        """

        def run(mask: Tuple[bool, bool, bool, bool]) -> float:
            # Run a synchronous worker on first device (use mp.Queue for internal call, avoid starting many processes)
            device_id = self.devices[0] if len(self.devices) > 0 else None
            q = mp.Queue()
            # Use same worker function to maintain logic consistency
            self._worker_run(
                rank=0,
                device_id=device_id,
                x_cpu=x_cpu,
                w_cpu=w_cpu,
                params=self.params,
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
            # Return median as robust estimate
            return float(torch.median(torch.tensor(errors, dtype=torch.float32)).item())

        return {
            "input_storage_error": run((True, False, False, False)),
            "weight_storage_error": run((False, True, False, False)),
            "accumulation_error": run((False, False, True, False)),
            "demote_error": run((False, False, False, True)),
        }