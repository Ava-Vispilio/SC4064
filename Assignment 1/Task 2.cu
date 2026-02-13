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

#define M 8192
#define N 8192


// Global thread index: tid = blockIdx.x * blockDim.x + threadIdx.x
// Mapping to (i,j): row-major => i = tid / N, j = tid % N
__global__ void matAdd_1D(const float *__restrict__ A,
                          const float *__restrict__ B,
                          float *__restrict__ C,
                          int rows, int cols)
{
    // Global linear index: which block * threads per block + thread in block
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Map linear index to matrix (row, col) in row-major order
    int i = tid / cols;
    int j = tid % cols;

    if (i < rows && j < cols)
        C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
}


// 2D Configuration: 2D grid with 2D blocks
__global__ void matAdd_2D(const float *__restrict__ A,
                          const float *__restrict__ B,
                          float *__restrict__ C,
                          int rows, int cols)
{
    // Row index:    i = blockIdx.y * blockDim.y + threadIdx.y
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    // Column index: j = blockIdx.x * blockDim.x + threadIdx.x
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols)
        C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
}

int main(void)
{
    size_t numElements = (size_t)M * N;
    size_t size = numElements * sizeof(float);

    // Allocate memory on the host
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Initialize A, B with random values in [0, 100]; C to 0
    for (size_t k = 0; k < numElements; k++) {
        h_A[k] = (float)(rand() / (double)RAND_MAX * 100.0);
        h_B[k] = (float)(rand() / (double)RAND_MAX * 100.0);
        h_C[k] = 0.0f;
    }

    // Allocate memory on the device
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // Copy A and B from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double flops_1D = 0.0, flops_2D = 0.0;
    float ms_1D = 0.0f, ms_2D = 0.0f;

    // ---------- 1D grid with 1D blocks ----------
    {
        const int BLOCK_SZ = 256;
        int totalThreads = M * N;
        int gridSize = (totalThreads + BLOCK_SZ - 1) / BLOCK_SZ;

        CUDA_CHECK(cudaEventRecord(start));
        matAdd_1D<<<gridSize, BLOCK_SZ>>>(d_A, d_B, d_C, M, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&ms_1D, start, stop));
        flops_1D = (double)numElements / (ms_1D / 1000.0);
        printf("1D config: grid %d, block %d, time %.3f ms, FLOPS %.3e\n",
               gridSize, BLOCK_SZ, ms_1D, flops_1D);
    }

    // ---------- 2D grid with 2D blocks ----------
    {
        const int BLOCK_X = 32;
        const int BLOCK_Y = 32;
        dim3 blockDim(BLOCK_X, BLOCK_Y);
        dim3 gridDim((N + BLOCK_X - 1) / BLOCK_X, (M + BLOCK_Y - 1) / BLOCK_Y);

        CUDA_CHECK(cudaEventRecord(start));
        matAdd_2D<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&ms_2D, start, stop));
        flops_2D = (double)numElements / (ms_2D / 1000.0);
        printf("2D config: grid (%d,%d), block (%d,%d), time %.3f ms, FLOPS %.3e\n",
               gridDim.x, gridDim.y, BLOCK_X, BLOCK_Y, ms_2D, flops_2D);
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    printf("\nComparison: 1D FLOPS = %.3e, 2D FLOPS = %.3e\n", flops_1D, flops_2D);
    return 0;
}
