# cuda-matmul-shared

**Tiled matrix multiplication using CUDA shared memory**  
A small example project demonstrating a tiled/shared-memory CUDA kernel for matrix multiplication (`matmul_shared.cu`). The project includes a CPU reference implementation for correctness checking and timing, kernel timing using CUDA events, and GFLOPS reporting.

---

## Features
- CUDA kernel using shared memory with configurable `TILE_WIDTH`.
- CPU reference (naive) multiply for correctness and time baseline.
- Automatic boundary handling for non-multiples of tile size.
- Simple error checking macro for CUDA calls.
- GFLOPS reporting (kernel-only).
- Command-line sizes and default sizes (512×512).

---

## Files
- `matmul_shared.cu` — main CUDA/host code (compile & run).
- `README.md` — this file.
- `Makefile` *(optional snippet provided below)*

---

## Requirements
- NVIDIA GPU with CUDA toolkit installed.
- `nvcc` available on PATH.
- Tested with CUDA toolkit versions supporting C++11/C++14.

---

## Build

Compile with `nvcc`:

```bash
nvcc -O3 matmul_shared.cu -o matmul_shared
