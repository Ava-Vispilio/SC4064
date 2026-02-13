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
#define K 8192
#define N 8192


// Matrix multiplication kernel: each thread computes one C(i,j).         
__global__ void matMul(const float *__restrict__ A,
                       const float *__restrict__ B,
                       float *__restrict__ C,
                       int M_dim, int K_dim, int N_dim)
{
    // Row and column of C (and thus which row of A, which column of B)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row i = blockIdx.y * blockDim.y + threadIdx.y   
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Col j = blockIdx.x * blockDim.x + threadIdx.x

    if (i >= M_dim || j >= N_dim)
        return;

    // Inner product: C(i,j) = sum_k A(i,k) * B(k,j)
    float sum = 0.0f;
    for (int k = 0; k < K_dim; k++)
        sum += A[i * K_dim + k] * B[k * N_dim + j];

    C[i * N_dim + j] = sum;
}

int main(void)
{
    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    // Allocate memory on the host
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Initialize A, B with random values in [0, 1]; C to 0
    for (size_t i = 0; i < (size_t)M * K; i++)
        h_A[i] = (float)(rand() / (double)RAND_MAX);
    for (size_t i = 0; i < (size_t)K * N; i++)
        h_B[i] = (float)(rand() / (double)RAND_MAX);
    for (size_t i = 0; i < (size_t)M * N; i++)
        h_C[i] = 0.0f;

    // Allocate memory on the device
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // Copy A and B to device; C is written by the kernel
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Total FLOPs for C = A*B: each C(i,j) does K muls + (K-1) adds ~ 2*K FLOPs; M*N elements
    double totalFlops = 2.0 * (double)M * (double)N * (double)K;

    // Task 4: Test at least three 2D block sizes (8×8, 16×16, 32×32)
    const int blockSizes[][2] = { {8, 8}, {16, 16}, {32, 32} };
    const int numConfigs = 3;

    for (int c = 0; c < numConfigs; c++) {
        int bx = blockSizes[c][0];
        int by = blockSizes[c][1];
        dim3 block(bx, by);
        dim3 grid((N + bx - 1) / bx, (M + by - 1) / by);

        CUDA_CHECK(cudaEventRecord(start));
        matMul<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        double sec = ms / 1000.0;
        double flops = totalFlops / sec;

        printf("Block (%d x %d), grid (%d x %d), time %.3f ms, FLOPS %.3e\n",
               bx, by, grid.x, grid.y, ms, flops);
    }

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
