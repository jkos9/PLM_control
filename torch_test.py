import numpy as np
import torch
import time

def benchmark_fft(shape=(1024, 1024), iterations=100):
    print(f"Benchmarking 2D FFT + Shift + Magnitude on shape {shape}")
    print(f"Iterations: {iterations}\n")

    # --- Setup ---
    # Simulate an 8-bit image converted to float32
    np_image = np.random.rand(*shape).astype(np.float32) * 255.0

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available! PyTorch will run on CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(device)}\n")

    # Move tensor to the target device (GPU)
    torch_image = torch.from_numpy(np_image).to(device)

    # ==========================================
    # 1. NumPy Benchmark (CPU)
    # ==========================================
    print("Running NumPy benchmark...")
    np_start = time.perf_counter()
    
    for _ in range(iterations):
        fft_complex = np.fft.fft2(np_image)
        fft_shifted = np.fft.fftshift(fft_complex)
        magnitude = np.abs(fft_shifted)
        _ = np.log1p(magnitude)
        
    np_end = time.perf_counter()
    np_avg_ms = ((np_end - np_start) / iterations) * 1000.0


    # ==========================================
    # 2. PyTorch Benchmark (GPU)
    # ==========================================
    print("Running PyTorch benchmark...")
    
    # WARMUP: Never measure the first few CUDA operations.
    for _ in range(10):
        t_fft = torch.fft.fft2(torch_image)
        t_shift = torch.fft.fftshift(t_fft)
        t_mag = torch.abs(t_shift)
        _ = torch.log1p(t_mag)
        
    if device.type == 'cuda':
        torch.cuda.synchronize() # Wait for GPU to finish warmup
        
    torch_start = time.perf_counter()
    
    for _ in range(iterations):
        t_fft = torch.fft.fft2(torch_image)
        t_shift = torch.fft.fftshift(t_fft)
        t_mag = torch.abs(t_shift)
        _ = torch.log1p(t_mag)
        
    if device.type == 'cuda':
        torch.cuda.synchronize() # MUST synchronize before stopping the clock!
        
    torch_end = time.perf_counter()
    torch_avg_ms = ((torch_end - torch_start) / iterations) * 1000.0


    # ==========================================
    # Results
    # ==========================================
    print("-" * 40)
    print(f"NumPy Average:   {np_avg_ms:.2f} ms per frame")
    print(f"PyTorch Average: {torch_avg_ms:.2f} ms per frame")
    
    if torch_avg_ms > 0:
        speedup = np_avg_ms / torch_avg_ms
        print(f"\nSpeedup: {speedup:.1f}x faster with PyTorch")

if __name__ == "__main__":
    # You can change the shape to match your camera's exact resolution
    benchmark_fft(shape=(500, 600), iterations=200)

