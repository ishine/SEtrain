import sys
import os
import argparse
import importlib
import torch
import time
from math import sqrt
from ptflops import get_model_complexity_info

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from models_new.modules.conv import set_calculate_macs_mode
except ImportError:
    print("Warning: Could not import set_calculate_macs_mode from models_new.modules.conv")
    set_calculate_macs_mode = None

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate model complexity and inference speed")
    parser.add_argument('module', type=str, help="Module path (e.g. models_new.gtcrn_prime_c)")
    parser.add_argument('--class_name', type=str, default="GTCRN", help="Model class name (default: GTCRN)")
    parser.add_argument('--input_length', type=int, default=160000, help="Input length for CPU benchmark (default: 160000)")
    parser.add_argument('--gpu_input_length', type=int, default=1600000, help="Input length for GPU benchmark (default: 1600000)")
    parser.add_argument('--runs', type=int, default=30, help="Number of runs for benchmark")
    return parser.parse_args()

def check_causality(model):
    print("Checking causality...")
    a = torch.randn(1, 16000)
    b = torch.randn(1, 16000)
    c = torch.randn(1, 16000)
    x1 = torch.cat([a, b], dim=1)
    x2 = torch.cat([a, c], dim=1)

    with torch.no_grad():
        # Handle potential multiple outputs or tuple returns
        y1 = model(x1)
        y2 = model(x2)
        
        if isinstance(y1, (tuple, list)):
            y1 = y1[0]
        if isinstance(y2, (tuple, list)):
            y2 = y2[0]

    # Check that the output corresponding to 'a' is identical in both cases
    # We allow a small margin at the boundary due to potential receptive field / padding issues
    # Original script used 256*2 = 512 samples margin
    margin = 512
    if y1.shape[-1] > margin:
        diff_common = (y1[..., :16000-margin] - y2[..., :16000-margin]).abs().max()
        print(f"Max difference in common segment (should be 0): {diff_common}")
    else:
        print("Output too short to check causality with margin.")
    
    # Verify that the second part is indeed different (sanity check)
    if y1.shape[-1] >= 16000:
        diff_distinct = (y1[..., 16000:] - y2[..., 16000:]).abs().max()
        print(f"Max difference in distinct segment (should be > 0): {diff_distinct}")

def benchmark_model(model: torch.nn.Module,
                    device: torch.device,
                    input_length: int = 16000,
                    runs: int = 50,
                    warmup: int = 10) -> tuple[float, float, float]:
    """
    Returns (mean_latency_seconds, mean_rtf, rtf_std)
    """
    model = model.to(device).eval()
    x = torch.randn(1, input_length, device=device)
    input_seconds = input_length / 16000.0

    print(f"Warming up on {device}...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    print(f"Running benchmark ({runs} runs)...")
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
    if len(times) > 1:
        sample_variance_latency = sum((t - mean_latency) ** 2 for t in times) / (len(times) - 1)
        rtf_std = sqrt(sample_variance_latency) / input_seconds
    else:
        rtf_std = float('nan')
    rtf = mean_latency / input_seconds
    return mean_latency, rtf, rtf_std

def main():
    args = get_args()
    
    try:
        module = importlib.import_module(args.module)
        ModelClass = getattr(module, args.class_name)
    except ImportError as e:
        print(f"Error importing module {args.module}: {e}")
        return
    except AttributeError:
        print(f"Error: Class {args.class_name} not found in {args.module}")
        return
    except Exception as e:
        print(f"Unexpected error during import: {e}")
        return

    print(f"\n======== Evaluating {args.module}.{args.class_name} ========")

    if set_calculate_macs_mode:
        set_calculate_macs_mode(True)
    
    try:
        model = ModelClass().eval()
        print("\n[Complexity Analysis]")
        flops, params = get_model_complexity_info(model, (16000,), as_strings=True,
                                                print_per_layer_stat=False, verbose=False, backend='aten')
        
        # Manual param count verification
        param_count = sum(p.numel() for p in model.parameters())
        print(f"MACs: {flops}")
        print(f"Params: {param_count/1e3:.2f} K")
    except Exception as e:
        print(f"Error calculating MACs/Params: {e}")

    print("\n[Causality Check]")
    try:
        check_causality(model)
    except Exception as e:
        print(f"Causality check failed: {e}")
    
    model = ModelClass().eval()
    
    # CPU Benchmark
    print("\n[Inference Time - CPU]")
    cpu_device = torch.device('cpu')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    
    try:
        cpu_latency_s, cpu_rtf, cpu_rtf_std = benchmark_model(
            model, 
            cpu_device, 
            input_length=args.input_length, 
            runs=args.runs, 
            warmup=5
        )
        print(f"Input duration: {args.input_length/16000:.2f} s")
        print(f"CPU mean latency: {cpu_latency_s*1000:.2f} ms | RTF: {cpu_rtf:.6f} +- {cpu_rtf_std:.6f}")
    except Exception as e:
        print(f"CPU Benchmark failed: {e}")
    
    if set_calculate_macs_mode:
        set_calculate_macs_mode(False)

    # GPU Benchmark
    if torch.cuda.is_available():
        print("\n[Inference Time - GPU]")
        try:
            gpu_device = torch.device('cuda')
            # For GPU, we might want to reinstantiate on GPU or move it
            model_gpu = ModelClass().to(gpu_device).eval()
            
            torch.backends.cudnn.benchmark = True
            gpu_latency_s, gpu_rtf, gpu_rtf_std = benchmark_model(
                model_gpu, 
                gpu_device, 
                input_length=args.gpu_input_length, 
                runs=max(100, args.runs * 2), 
                warmup=10
            )
            print(f"Input duration: {args.gpu_input_length/16000:.2f} s")
            print(f"GPU mean latency: {gpu_latency_s*1000:.2f} ms | RTF: {gpu_rtf:.6f} +- {gpu_rtf_std:.6f}")
        except Exception as e:
             print(f"GPU Benchmark failed: {e}")
    else:
        print("\n[Inference Time - GPU]")
        print("CUDA not available; skipping GPU benchmark.")

if __name__ == "__main__":
    main()
