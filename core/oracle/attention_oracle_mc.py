# precision_estimation/core/oracle/attention_oracle_mc.py
"""
DataAwareMCAttentionOracle

Data-aware Monte Carlo Attention error oracle
- Supports multi-GPU parallel processing
- Supports element-wise ULP noise simulation
- Multi-stage error analysis for attention: QK^T -> Scale -> Softmax -> @V
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


class DataAwareMCAttentionOracle:
    """
    Data-aware Monte Carlo Attention error oracle
    
    Attention flow: Q@K^T -> scale -> mask -> softmax -> @V
    Error sources: input_storage(Q,K,V) + matmul1_accum + scaling + softmax_numerical + matmul2_accum + output_demote
    """

    def __init__(
        self,
        strategy: PrecisionStrategy,
        attention_params: Dict[str, Any],
        num_mc_samples: int = 512,
        quantile: float = 0.999,
        safety_factor: float = 1.10,
        seeded: bool = True,
        devices: Optional[List[int]] = None,
        enable_noise_input: bool = True,
        enable_noise_matmul1: bool = True,
        enable_noise_scaling: bool = True,
        enable_noise_softmax: bool = True,
        enable_noise_matmul2: bool = True,
        enable_noise_output: bool = True,
    ):
        self.strategy = strategy
        self.params = {
            "scale": attention_params.get("scale", 1.0),
            "is_causal": attention_params.get("is_causal", True),
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

        # Noise switches - attention-specific multi-stage
        self.enable_noise_input = enable_noise_input
        self.enable_noise_matmul1 = enable_noise_matmul1
        self.enable_noise_scaling = enable_noise_scaling
        self.enable_noise_softmax = enable_noise_softmax
        self.enable_noise_matmul2 = enable_noise_matmul2
        self.enable_noise_output = enable_noise_output

    @staticmethod
    @torch.inference_mode()
    def _compute_reference_on_device(
        q_cpu: torch.Tensor, 
        k_cpu: torch.Tensor, 
        v_cpu: torch.Tensor, 
        mask_cpu: torch.Tensor,
        device: torch.device, 
        params: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute FP64 high-precision reference
        """
        q64 = q_cpu.to(device=device, dtype=torch.float64)
        k64 = k_cpu.to(device=device, dtype=torch.float64)
        v64 = v_cpu.to(device=device, dtype=torch.float64)
        mask64 = mask_cpu.to(device=device, dtype=torch.bool)
        
        scale = params["scale"]
        is_causal = params["is_causal"]
        
        # Attention computation
        scores = torch.matmul(q64, k64.transpose(-2, -1)) * scale
        
        if is_causal:
            scores = scores.masked_fill(~mask64, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v64)
        
        return output.to(dtype=torch.float32)

    @staticmethod
    def _worker_run(
        rank: int,
        device_id: Optional[int],
        q_cpu: torch.Tensor,
        k_cpu: torch.Tensor,
        v_cpu: torch.Tensor,
        mask_cpu: torch.Tensor,
        params: Dict[str, Any],
        strategy: PrecisionStrategy,
        num_local: int,
        noise_mask: Tuple[bool, bool, bool, bool, bool, bool],
        seed_base: int,
        return_queue: mp.Queue,
    ):
        """
        Attention worker
        noise_mask: (input, matmul1, scaling, softmax, matmul2, output)
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
            y_ref = DataAwareMCAttentionOracle._compute_reference_on_device(
                q_cpu, k_cpu, v_cpu, mask_cpu, device, params
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
                q = q_cpu.to(device=device, dtype=torch.float32)
                k = k_cpu.to(device=device, dtype=torch.float32)
                v = v_cpu.to(device=device, dtype=torch.float32)
                mask = mask_cpu.to(device=device, dtype=torch.bool)

                # Storage precision quantization
                q_q = q.to(dtype=strategy.input_dtype).to(dtype=torch.float32)
                k_q = k.to(dtype=strategy.input_dtype).to(dtype=torch.float32)
                v_q = v.to(dtype=strategy.input_dtype).to(dtype=torch.float32)

                # Promote to compute precision
                q_c = q_q.to(dtype=strategy.compute_dtype)
                k_c = k_q.to(dtype=strategy.compute_dtype)
                v_c = v_q.to(dtype=strategy.compute_dtype)

                # ---- Input noise ----
                if noise_mask[0]:
                    ulp_q = ulp_like(q_c, strategy.compute_dtype).to(device=device)
                    ulp_k = ulp_like(k_c, strategy.compute_dtype).to(device=device)
                    ulp_v = ulp_like(v_c, strategy.compute_dtype).to(device=device)
                    
                    r_q = torch.rand(q_c.shape, generator=g, device=device, dtype=q_c.dtype)
                    r_k = torch.rand(k_c.shape, generator=g, device=device, dtype=k_c.dtype)
                    r_v = torch.rand(v_c.shape, generator=g, device=device, dtype=v_c.dtype)
                    
                    q_c = q_c + (r_q - 0.5) * ulp_q
                    k_c = k_c + (r_k - 0.5) * ulp_k
                    v_c = v_c + (r_v - 0.5) * ulp_v

                # ---- First matrix multiplication: Q@K^T ----
                scores = torch.matmul(q_c, k_c.transpose(-2, -1))
                
                # matmul1 accumulation noise
                if noise_mask[1]:
                    ulp_scores = ulp_like(scores, strategy.compute_dtype).to(device=device)
                    r = torch.rand(scores.shape, generator=g, device=device, dtype=scores.dtype)
                    scores = scores + (r - 0.5) * ulp_scores

                # ---- Scaling ----
                scale = params["scale"]
                scores = scores * scale
                
                # Scaling noise
                if noise_mask[2]:
                    ulp_scaled = ulp_like(scores, strategy.compute_dtype).to(device=device)
                    r = torch.rand(scores.shape, generator=g, device=device, dtype=scores.dtype)
                    scores = scores + (r - 0.5) * ulp_scaled

                # ---- Apply mask ----
                if params["is_causal"]:
                    scores = scores.masked_fill(~mask, float('-inf'))

                # ---- Softmax computation ----
                attn_weights = F.softmax(scores, dim=-1)
                
                # Softmax numerical error
                if noise_mask[3]:
                    ulp_attn = ulp_like(attn_weights, strategy.compute_dtype).to(device=device)
                    r = torch.rand(attn_weights.shape, generator=g, device=device, dtype=attn_weights.dtype)
                    attn_weights = attn_weights + (r - 0.5) * ulp_attn
                    # Renormalize to maintain probability properties
                    attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-12)

                # ---- Second matrix multiplication: Attention@V ----
                output = torch.matmul(attn_weights, v_c)
                
                # matmul2 accumulation noise
                if noise_mask[4]:
                    ulp_output = ulp_like(output, strategy.compute_dtype).to(device=device)
                    r = torch.rand(output.shape, generator=g, device=device, dtype=output.dtype)
                    output = output + (r - 0.5) * ulp_output

                # ---- Output demotion ----
                y_out = output.to(dtype=strategy.output_dtype).to(dtype=torch.float32)
                if noise_mask[5]:
                    ulp_final = ulp_like(y_out, strategy.output_dtype).to(device=device)
                    r = torch.rand(y_out.shape, generator=g, device=device, dtype=y_out.dtype)
                    y_out = y_out + (r - 0.5) * ulp_final

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
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: torch.Tensor
    ) -> OracleResult:
        """
        Predict Attention error bound
        """
        q_cpu = q.detach().contiguous().cpu()
        k_cpu = k.detach().contiguous().cpu()
        v_cpu = v.detach().contiguous().cpu()
        mask_cpu = mask.detach().contiguous().cpu()

        noise_mask = (
            self.enable_noise_input,
            self.enable_noise_matmul1,
            self.enable_noise_scaling,
            self.enable_noise_softmax,
            self.enable_noise_matmul2,
            self.enable_noise_output,
        )

        if len(self.devices) == 0:
            q = mp.Queue()
            self._worker_run(
                rank=0, device_id=None, q_cpu=q_cpu, k_cpu=k_cpu, v_cpu=v_cpu, mask_cpu=mask_cpu,
                params=self.params, strategy=self.strategy, num_local=self.num_mc_samples,
                noise_mask=noise_mask, seed_base=1234 if self.seeded else int(time.time()),
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
                        rank, dev, q_cpu, k_cpu, v_cpu, mask_cpu, self.params, self.strategy, per,
                        noise_mask, 1234 if self.seeded else int(time.time()), q,
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

        comp = self._estimate_components(q_cpu, k_cpu, v_cpu, mask_cpu, num_samples=min(128, max(8, self.num_mc_samples // 8)))

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
        q_cpu: torch.Tensor, 
        k_cpu: torch.Tensor, 
        v_cpu: torch.Tensor, 
        mask_cpu: torch.Tensor,
        num_samples: int
    ) -> Dict[str, float]:
        """
        Estimate error components for each stage of attention
        """
        def run(mask: Tuple[bool, bool, bool, bool, bool, bool]) -> float:
            device_id = self.devices[0] if len(self.devices) > 0 else None
            q = mp.Queue()
            self._worker_run(
                rank=0, device_id=device_id, q_cpu=q_cpu, k_cpu=k_cpu, v_cpu=v_cpu, mask_cpu=mask_cpu,
                params=self.params, strategy=self.strategy, num_local=num_samples,
                noise_mask=mask, seed_base=4321 if self.seeded else int(time.time()),
                return_queue=q,
            )
            _, errors, err_msg = q.get()
            if err_msg:
                raise RuntimeError(f"Component worker error: {err_msg}")
            if len(errors) == 0:
                return 0.0
            return float(torch.median(torch.tensor(errors, dtype=torch.float32)).item())

        return {
            "input_storage_error": run((True, False, False, False, False, False)),
            "matmul1_accumulation_error": run((False, True, False, False, False, False)),
            "scaling_error": run((False, False, True, False, False, False)),
            "softmax_numerical_error": run((False, False, False, True, False, False)),
            "matmul2_accumulation_error": run((False, False, False, False, True, False)),
            "output_demote_error": run((False, False, False, False, False, True)),
        }