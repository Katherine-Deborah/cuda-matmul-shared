// matmul_shared.cu
// Compile: nvcc -O3 matmul_shared.cu -o matmul_shared
// Run:     ./matmul_shared [M K N]   (defaults below if no args)

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define TILE_WIDTH 16  // try 16 or 32 depending on your GPU

// Simple CUDA error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Kernel: tiled matrix multiply using shared memory
__global__ void matMulShared(const float* A, const float* B, float* C,
    int M, int K, int N) {
    // Block/Thread indices
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Row and column of C computed by this thread
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Accumulator for the dot product
    float Cvalue = 0.0f;

    // Number of tiles (over K dimension)
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    // Shared memory for a tile of A and B
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    for (int t = 0; t < numTiles; ++t) {
        // Global indices to read
        int aCol = t * TILE_WIDTH + tx;
        int bRow = t * TILE_WIDTH + ty;

        // Load from global memory to shared memory with boundary checks
        As[ty][tx] = (Row < M && aCol < K) ? A[Row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && Col < N) ? B[bRow * N + Col] : 0.0f;

        // Make sure all threads loaded their elements
        __syncthreads();

        // Multiply the two tiles
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        // Wait before next tile load (avoid racing next loads)
        __syncthreads();
    }

    // Write result back to global memory
    if (Row < M && Col < N) {
        C[Row * N + Col] = Cvalue;
    }
}

// CPU reference (naive) matmul: C = A * B
void cpuMatMul(const std::vector<float>& A, const std::vector<float>& B,
    std::vector<float>& C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    // Default sizes (M x K) * (K x N) = M x N
    int M = 512, K = 512, N = 512;
    if (argc >= 4) {
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        N = atoi(argv[3]);
    }
    std::cout << "Matrix sizes: A(" << M << "x" << K << ") B(" << K << "x" << N << ") -> C(" << M << "x" << N << ")\n";

    // Host allocations
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);

    // Initialize A and B with random floats
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : h_A) x = dist(rng);
    for (auto& x : h_B) x = dist(rng);

    // CPU reference (optional; may be slow for large sizes)
    std::cout << "Running CPU reference multiply (this may be slow)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    cpuMatMul(h_A, h_B, h_C_ref, M, K, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "CPU time: " << cpuTime << " s\n";

    // Device allocations
    float* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
    size_t sizeA = sizeof(float) * M * K;
    size_t sizeB = sizeof(float) * K * N;
    size_t sizeC = sizeof(float) * M * N;
    gpuErrchk(cudaMalloc(&d_A, sizeA));
    gpuErrchk(cudaMalloc(&d_B, sizeB));
    gpuErrchk(cudaMalloc(&d_C, sizeC));

    // Copy to device
    gpuErrchk(cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice));

    // Kernel launch config
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    std::cout << "Launching kernel: grid(" << dimGrid.x << "," << dimGrid.y << ") block(" << dimBlock.x << "," << dimBlock.y << ")\n";

    // Warmup + timed runs
    const int repeats = 10;
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start));

    // run kernel 'repeats' times (to stabilize timing)
    for (int i = 0; i < repeats; ++i) {
        matMulShared << <dimGrid, dimBlock >> > (d_A, d_B, d_C, M, K, N);
    }

    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    float ms = 0.0f;
    gpuErrchk(cudaEventElapsedTime(&ms, start, stop));
    double gpuTimeSec = (ms / 1000.0) / repeats; // average per run
    std::cout << "Average GPU kernel time: " << gpuTimeSec << " s\n";

    // Copy result back
    gpuErrchk(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));

    // Compare correctness (max absolute error)
    double maxErr = 0.0;
    double sumErr = 0.0;
    for (size_t i = 0; i < h_C.size(); ++i) {
        double err = fabs(h_C_ref[i] - h_C[i]);
        sumErr += err * err;
        if (err > maxErr) maxErr = err;
    }
    std::cout << "Max absolute error: " << maxErr << "\n";
    std::cout << "RMS error: " << sqrt(sumErr / (h_C.size())) << "\n";

    // Compute GFLOPS: 2*M*N*K floating point ops
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = flops / (gpuTimeSec * 1e9);
    std::cout << "Approx GFLOPS (kernel only): " << gflops << "\n";

    // Free device memory
    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_B));
    gpuErrchk(cudaFree(d_C));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));

    return 0;
}
