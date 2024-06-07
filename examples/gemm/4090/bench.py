import torch
from torch import nn
import sys
import os
from torch.cuda import Event
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)
sys.path.append('build/lib.linux-x86_64-cpython-312')
import simple_gemm as tk

def flops_gemm(m, n, k):
    return 2 * m * n * k

def efficiency(flop, time):
    # calculate in teraflops
    flop = flop / 1e12
    time = time / 1e6
    return flop / time

def measure_gemm_performance(m, n, k):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == 'cuda', "CUDA not available"
    print("Using device:", device)

    # Generate random data for A and B matrices
    A = torch.randn(m, k, device=device, dtype=torch.float16).contiguous()
    B = torch.randn(k, n, device=device, dtype=torch.float16).contiguous()
    
    C = torch.zeros(m, n, device=device, dtype=torch.float16).contiguous()
    
    torch.cuda.synchronize()

    # Warm up custom kernel
    for _ in range(10):
        tk.gemm(A, B, C)
    
    torch.cuda.synchronize()
    
    # Measure custom kernel
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    
    for i in range(100):
        start_events[i].record()
        torch.cuda.synchronize()
        tk.gemm(A, B, C)
        torch.cuda.synchronize()
        end_events[i].record()
    
    torch.cuda.synchronize()
    custom_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    
    custom_time_us = np.mean(custom_times) * 1000
    custom_tflops = efficiency(flops_gemm(m, n, k), custom_time_us)
    
    # Warm up PyTorch matmul
    for _ in range(10):
        torch.mm(A, B, out=C)
    
    torch.cuda.synchronize()
    
    # Measure PyTorch matmul
    for i in range(100):
        start_events[i].record()
        torch.cuda.synchronize()
        torch.mm(A, B, out=C)
        torch.cuda.synchronize()
        end_events[i].record()
    
    torch.cuda.synchronize()
    pytorch_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    
    pytorch_time_us = np.mean(pytorch_times) * 1000
    pytorch_tflops = efficiency(flops_gemm(m, n, k), pytorch_time_us)
    
    print(f"M = {m}, N = {n}, K = {k}")
    print(f"Custom GEMM - Average time taken: {custom_time_us:.2f} us, Efficiency: {custom_tflops:.2f} TFLOPS")
    print(f"PyTorch GEMM - Average time taken: {pytorch_time_us:.2f} us, Efficiency: {pytorch_tflops:.2f} TFLOPS")
    print(f"______________________________________________________")

    return (m, n, k, custom_time_us, custom_tflops, pytorch_time_us, pytorch_tflops)

# Test configurations
configs = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192)
]

results = []

for config in configs:
    results.append(measure_gemm_performance(*config))

# Extract results for plotting
configs = [f"{m}x{n}x{k}" for (m, n, k, _, _, _, _) in results]
custom_tflops = [custom_tflops for (_, _, _, _, custom_tflops, _, _) in results]
pytorch_tflops = [pytorch_tflops for (_, _, _, _, _, _, pytorch_tflops) in results]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(configs, custom_tflops, label='Custom GEMM', marker='o')
plt.plot(configs, pytorch_tflops, label='PyTorch GEMM', marker='x')
plt.xlabel('Configuration (MxNxK)')
plt.ylabel('Efficiency (TFLOPS)')
plt.title('GEMM Performance Comparison')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('gemm_performance_comparison.png')
plt.show()
