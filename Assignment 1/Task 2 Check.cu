/*
 * Task 2 Check: Prints which thread loads which index from A, B, and C,
 * and from which memory address. Uses a small 4x4 matrix so output is readable.
 * Helps compare 1D vs 2D grid/block access order and coalescing.
 */

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

#define ROWS 4
#define COLS 4
#define N (ROWS * COLS)

// Only the first MAX_PRINT threads print; avoids flooding the console
#define MAX_PRINT 16


// 1D kernel: same indexing as Task 2 matAdd_1D, plus printf to show access
__global__ void matAdd_1D_verbose(const float *A, const float *B, float *C,
                                  int rows, int cols)
{
    // Global linear index and mapping to (i, j) as in 1D config
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / cols;
    int j = tid % cols;

    if (tid < rows * cols)
        C[tid] = A[tid] + B[tid];

    // Print from first MAX_PRINT threads only: which thread, which index,
    // which value from A/B, and which device address (for A, B, C)
    if (tid < MAX_PRINT && tid < rows * cols) {
        int idx = i * cols + j;
        unsigned long addr_A = (unsigned long)(A + idx);
        unsigned long addr_B = (unsigned long)(B + idx);
        unsigned long addr_C = (unsigned long)(C + idx);
        printf("[1D] blockIdx.x=%d threadIdx.x=%d -> tid=%d (i=%d,j=%d) idx=%d\n"
               "     A[idx]=%g loaded from A+%d (addr %lu)\n"
               "     B[idx]=%g loaded from B+%d (addr %lu)\n"
               "     C[idx]=A+B written to C+%d (addr %lu)\n\n",
               blockIdx.x, threadIdx.x, tid, i, j, idx,
               A[idx], idx, addr_A,
               B[idx], idx, addr_B,
               idx, addr_C);
    }
}


// 2D kernel: same indexing as Task 2 matAdd_2D, plus printf to show access
__global__ void matAdd_2D_verbose(const float *A, const float *B, float *C,
                                  int rows, int cols)
{
    // Row and column from 2D block/thread indices
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * cols + j;

    if (i < rows && j < cols)
        C[idx] = A[idx] + B[idx];

    // Print only from the first block so we can compare order with 1D
    if (blockIdx.x == 0 && blockIdx.y == 0 && i < rows && j < cols) {
        unsigned long addr_A = (unsigned long)(A + idx);
        unsigned long addr_B = (unsigned long)(B + idx);
        unsigned long addr_C = (unsigned long)(C + idx);
        printf("[2D] blockIdx=(%d,%d) threadIdx=(%d,%d) -> (i=%d,j=%d) idx=%d\n"
               "     A[idx]=%g loaded from A+%d (addr %lu)\n"
               "     B[idx]=%g loaded from B+%d (addr %lu)\n"
               "     C[idx]=A+B written to C+%d (addr %lu)\n\n",
               blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, i, j, idx,
               A[idx], idx, addr_A,
               B[idx], idx, addr_B,
               idx, addr_C);
    }
}

int main(void)
{
    size_t size = N * sizeof(float);

    // Allocate memory on the host
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Initialize with simple values so printed numbers are easy to follow
    // (A[k]=k+1, B[k]=k+10; C will be overwritten by kernels)
    for (int k = 0; k < N; k++) {
        h_A[k] = (float)(k + 1);
        h_B[k] = (float)(k + 10);
        h_C[k] = 0.0f;
    }

    // Allocate memory on the device
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // Copy A and B to device; C is written by the kernels
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Run 1D kernel: one block of N threads so thread order = tid 0,1,...,15
    printf("===== 1D grid, 1D blocks =====\n");
    matAdd_1D_verbose<<<1, N>>>(d_A, d_B, d_C, ROWS, COLS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run 2D kernel: 2x2 grid of 2x2 blocks; print only from first block
    printf("===== 2D grid, 2D blocks (first block only) =====\n");
    matAdd_2D_verbose<<<dim3(2, 2), dim3(2, 2)>>>(d_A, d_B, d_C, ROWS, COLS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
