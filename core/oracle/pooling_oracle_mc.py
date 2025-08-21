import os, math, time, traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from core.config.precision_strategy import PrecisionStrategy, ulp_like

@dataclass
class OracleResult:
    predicted_bound: float
    quantile: float
    safety_factor: float
    sample_errors: List[float]
    component_estimates: Dict[str,float]
    meta: Dict[str,Any]

class DataAwareMCPoolingOracle:
    """
    Pooling Data-Aware Monte Carlo Oracle
    """

    def __init__(
        self,
        strategy: PrecisionStrategy,
        pool_params: Dict[str,Any],
        num_mc_samples: int = 512,
        quantile: float = 0.999,
        safety_factor: float = 1.10,
        seeded: bool = True,
        devices: Optional[List[int]] = None,
        enable_noise_input: bool = True,
        enable_noise_accum: bool = True,
    ):
        self.strategy = strategy
        self.pool_params = pool_params
        self.num_mc_samples = int(num_mc_samples)
        self.quantile = float(quantile)
        self.safety_factor = float(safety_factor)
        self.seeded = seeded

        if devices is None:
            if torch.cuda.is_available():
                devices = list(range(torch.cuda.device_count()))
            else:
                devices = []
        self.devices = devices

        self.enable_noise_input = enable_noise_input
        self.enable_noise_accum = enable_noise_accum

    @staticmethod
    @torch.inference_mode()
    def _compute_reference_on_device(x_cpu: torch.Tensor, device: torch.device, pool_params: Dict[str,Any]) -> torch.Tensor:
        """Compute high-precision reference output on specified device"""
        x64 = x_cpu.to(device=device,dtype=torch.float64)
        if pool_params["pool_type"] == "max":
            y64 = F.max_pool2d(x64, kernel_size=pool_params["kernel_size"], stride=pool_params["stride"])
        elif pool_params["pool_type"] == "avg":
            y64 = F.avg_pool2d(x64, kernel_size=pool_params["kernel_size"], stride=pool_params["stride"])
        else:
            raise ValueError(f"Unknown pool_type {pool_params['pool_type']}")
        return y64.to(dtype=torch.float32)

    @staticmethod
    def _worker_run(
        rank:int,
        device_id:Optional[int],
        x_cpu:torch.Tensor,
        pool_params:Dict[str,Any],
        strategy:PrecisionStrategy,
        num_local:int,
        noise_mask:Tuple[bool,bool],
        seed_base:int,
        return_queue:mp.Queue
    ):
        """Worker function for parallel Monte Carlo sampling"""
        try:
            torch.set_num_threads(max(1, os.cpu_count()//8))
            device = torch.device(f"cuda:{device_id}") if device_id is not None and torch.cuda.is_available() else torch.device("cpu")
            y_ref = DataAwareMCPoolingOracle._compute_reference_on_device(x_cpu, device, pool_params)

            errors: List[float] = []

            for i in range(num_local):
                g = torch.Generator(device=device) if device.type=="cuda" else torch.Generator()
                g.manual_seed(seed_base + i)

                x = x_cpu.to(device=device,dtype=torch.float32)
                x_q = x.to(dtype=strategy.input_dtype).to(dtype=torch.float32)
                x_c = x_q.to(dtype=strategy.compute_dtype)

                # Add input noise if enabled
                if noise_mask[0]:
                    ulp_x = ulp_like(x_c, strategy.compute_dtype).to(device=device)
                    r = torch.rand(x_c.shape, generator=g, device=device, dtype=x_c.dtype)
                    x_c = x_c + (r-0.5)*ulp_x

                # Perform pooling operation
                if pool_params["pool_type"]=="max":
                    y_c = F.max_pool2d(x_c, kernel_size=pool_params["kernel_size"], stride=pool_params["stride"])
                elif pool_params["pool_type"]=="avg":
                    y_c = F.avg_pool2d(x_c, kernel_size=pool_params["kernel_size"], stride=pool_params["stride"])

                # Add accumulation noise if enabled
                if noise_mask[1]:
                    ulp_y = ulp_like(y_c, strategy.compute_dtype).to(device=device)
                    r = torch.rand(y_c.shape, generator=g, device=device, dtype=y_c.dtype)
                    y_c = y_c + (r-0.5)*ulp_y

                y_out = y_c.to(dtype=strategy.output_dtype).to(dtype=torch.float32)
                err = (y_out - y_ref).abs().max().item()
                errors.append(err)

            return_queue.put((rank, errors, None))

        except Exception as e:
            tb = traceback.format_exc()
            return_queue.put((rank, [], f"{repr(e)}\n{tb}"))

    def predict_error_bound(self, x:torch.Tensor) -> OracleResult:
        """Predict error bound using Monte Carlo sampling"""
        x_cpu = x.detach().contiguous().cpu()
        noise_mask = (self.enable_noise_input, self.enable_noise_accum)

        if len(self.devices)==0:
            # Single-threaded execution
            q = mp.Queue()
            self._worker_run(0,None,x_cpu,self.pool_params,self.strategy,self.num_mc_samples,noise_mask,1234,q)
            _, errors, err_msg = q.get()
            if err_msg: raise RuntimeError(f"Worker error: {err_msg}")
            all_errors = errors
        else:
            # Multi-device parallel execution
            per = math.ceil(self.num_mc_samples/len(self.devices))
            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            procs = []
            for rank, dev in enumerate(self.devices):
                p = ctx.Process(target=self._worker_run,args=(rank,dev,x_cpu,self.pool_params,self.strategy,per,noise_mask,1234,q))
                p.daemon = True
                p.start()
                procs.append(p)

            all_errors: List[float] = []
            for _ in range(len(procs)):
                _, errors, err_msg = q.get()
                if err_msg: raise RuntimeError(f"Worker error: {err_msg}")
                all_errors.extend(errors)

            for p in procs:
                p.join(timeout=60)
            for p in procs:
                if p.is_alive(): p.terminate()
            all_errors = all_errors[:self.num_mc_samples]

        if len(all_errors)==0: all_errors=[0.0]
        errs_tensor = torch.tensor(all_errors,dtype=torch.float32)
        qv = float(torch.quantile(errs_tensor,torch.tensor(self.quantile)))
        predicted = qv*self.safety_factor

        return OracleResult(
            predicted_bound=predicted,
            quantile=self.quantile,
            safety_factor=self.safety_factor,
            sample_errors=all_errors,
            component_estimates={}, # Can implement component estimation similar to conv2d if needed
            meta={"num_samples":len(all_errors),"devices":self.devices,"strategy":str(self.strategy),"noise_mask":noise_mask}
        )