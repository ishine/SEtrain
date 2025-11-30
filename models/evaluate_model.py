import gtcrn_dynamic_irm as module
from gtcrn_dynamic_irm import GTCRN
import torch
import time
from math import sqrt
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

module.calculate_macs_mode = True
model = GTCRN().eval()

"""complexity count"""
from ptflops import get_model_complexity_info
flops, params = get_model_complexity_info(model, (16000,), as_strings=True,
                                        print_per_layer_stat=True, verbose=False, backend='aten')
params = 0
for p in model.parameters():
    params += p.numel()
print(flops, params/1e3, "K")

"""causality check"""
a = torch.randn(1, 16000)
b = torch.randn(1, 16000)
c = torch.randn(1, 16000)
x1 = torch.cat([a, b], dim=1)
x2 = torch.cat([a, c], dim=1)

y1 = model(x1)[0]
y2 = model(x2)[0]

print("Causality check (first max diff should be 0):")
print((y1[:16000-256*2] - y2[:16000-256*2]).abs().max())
print((y1[16000:] - y2[16000:]).abs().max())

"""inference time benchmarking (CPU and GPU)
Real-time factor (RTF) = inference_duration / input_duration.
Input length 16000 samples corresponds to 1 second of audio.
"""

def benchmark_model(model: torch.nn.Module,
                    device: torch.device,
                    input_length: int = 16000,
                    runs: int = 50,
                    warmup: int = 10) -> tuple[float, float, float]:
    """
    Returns (mean_latency_seconds, mean_rtf, rtf_std)
    - mean_latency_seconds: average wall-clock latency per run (s)
    - mean_rtf: average real-time factor (dimensionless)
    - rtf_std: sample standard deviation of RTF across runs
    """
    model = model.to(device).eval()
    x = torch.randn(1, input_length, device=device)
    input_seconds = input_length / 16000.0

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    mean_latency = sum(times) / len(times)
    # Sample variance of latency (unbiased): sum((x - mean)^2) / (n - 1)
    if len(times) > 1:
        sample_variance_latency = sum((t - mean_latency) ** 2 for t in times) / (len(times) - 1)
        # Since RTF = latency / input_seconds, std(RTF) = std(latency) / input_seconds
        rtf_std = sqrt(sample_variance_latency) / input_seconds
    else:
        rtf_std = float('nan')
    rtf = mean_latency / input_seconds
    return mean_latency, rtf, rtf_std


print("\n==== Inference time (CPU) ====")
cpu_device = torch.device('cpu')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cpu_latency_s, cpu_rtf, cpu_rtf_std = benchmark_model(model, cpu_device, input_length=160000, runs=30, warmup=5)
print(f"CPU mean latency: {cpu_latency_s*1000:.2f} ms | RTF: {cpu_rtf:.3f}+-{cpu_rtf_std:.6f}")

# if torch.cuda.is_available():
#     print("\n==== Inference time (GPU) ====")
#     # For GPU inference, disable MACs counting mode and build a fresh model on CUDA
#     module.calculate_macs_mode = False
#     model_gpu = GTCRN().eval()
#     gpu_device = torch.device('cuda')
#     # Optional: cudnn autotune for performance stability
#     torch.backends.cudnn.benchmark = True
#     gpu_latency_s, gpu_rtf, gpu_rtf_std = benchmark_model(model_gpu, gpu_device, input_length=1600000, runs=150, warmup=10)
#     print(f"GPU mean latency: {gpu_latency_s*1000:.2f} ms | RTF: {gpu_rtf:.6f}+-{gpu_rtf_std:.6f}")
# else:
#     print("\nCUDA not available; skipping GPU benchmark.")