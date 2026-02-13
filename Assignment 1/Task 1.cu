#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error Checking Macro (from Week 2)
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (at %s:%d)\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit((int)err); \
    } \
} while(0)

#define N       (1ULL << 30)   // vector length 2^30 
#define BLOCK_SZ 32           // Specify case here!

// Kernel
__global__ void vecAdd(const float *__restrict__ A,
                       const float *__restrict__ B,
                       float *__restrict__ C,
                       int n)
{
    // In 1D array, index = which block x size of block + which thread in block
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only add if tid is within the array bounds
    if (tid < n)
        C[tid] = A[tid] + B[tid];
}

int main(void)
{
    // Allocate memory on the host
    size_t size = N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Initialize A, B with random values in [0, 100]
    for (size_t i = 0; i < N; i++) {
        h_A[i] = (float)(rand() / (double)RAND_MAX * 100.0);
        h_B[i] = (float)(rand() / (double)RAND_MAX * 100.0);
    }

    // Allocate memory on the device + error checking
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Calculate grid size
    int gridSize = (int)((N + BLOCK_SZ - 1) / BLOCK_SZ);

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vecAdd<<<gridSize, BLOCK_SZ>>>(d_A, d_B, d_C, (int)N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time and FLOPS
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float sec = ms / 1000.0f;
    double flops = (double)N / sec;
    printf("Block size %d, grid size %d, time %.3f ms, FLOPS %.3e\n",
           BLOCK_SZ, gridSize, ms, flops);

    // Destroy timing events       
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}