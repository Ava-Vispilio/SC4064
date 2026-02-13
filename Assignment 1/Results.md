# Assignment 1 Report

## Problem 1

### 32
* Run 1: `Block size 64, grid size 16777216, time 15.697 ms, FLOPS 6.840e+10`
* Run 2: `Block size 32, grid size 33554432, time 28.240 ms, FLOPS 3.802e+10`
* Run 3: `Block size 32, grid size 33554432, time 28.688 ms, FLOPS 3.743e+10`
* Run 4: `Block size 32, grid size 33554432, time 26.705 ms, FLOPS 4.021e+10`
* Run 5: `Block size 32, grid size 33554432, time 28.943 ms, FLOPS 3.710e+10`

Average of about ~3.7–4.0e+10 FLOPS and ~28 ms

### 64
* Run 1: `Block size 64, grid size 16777216, time 15.674 ms, FLOPS 6.851e+10`
* Run 2: `Block size 64, grid size 16777216, time 16.206 ms, FLOPS 6.625e+10`
* Run 3: `Block size 64, grid size 16777216, time 18.020 ms, FLOPS 5.959e+10`
* Run 4: `Block size 64, grid size 16777216, time 16.431 ms, FLOPS 6.535e+10`
* Run 5: `Block size 64, grid size 16777216, time 15.697 ms, FLOPS 6.840e+10`

Average of about ~6.5–6.9e+10 FLOPS and ~16 ms

### 128
* Run 1: `Block size 128, grid size 8388608, time 35.725 ms, FLOPS 3.006e+10`
* Run 2: `Block size 128, grid size 8388608, time 11.344 ms, FLOPS 9.465e+10`
* Run 3: `Block size 128, grid size 8388608, time 11.344 ms, FLOPS 9.465e+10`
* Run 4: `Block size 128, grid size 8388608, time 13.836 ms, FLOPS 7.761e+10`
* Run 5: `Block size 128, grid size 8388608, time 12.313 ms, FLOPS 8.720e+10`

Average of about ~7.7–9.5e+10 FLOPS and ~12 ms

### 256 
* Run 1: `Block size 256, grid size 4194304, time 21.873 ms, FLOPS 4.909e+10`
* Run 2: `Block size 256, grid size 4194304, time 13.648 ms, FLOPS 7.867e+10`
* Run 3: `Block size 256, grid size 4194304, time 13.415 ms, FLOPS 8.004e+10`
* Run 4: `Block size 256, grid size 4194304, time 11.533 ms, FLOPS 9.310e+10`
* Run 5: `Block size 256, grid size 4194304, time 13.824 ms, FLOPS 7.767e+10`

Average of about ~7.7–9.3e+10 FLOPS and ~13 ms

There are a few outliers in the timings (e.g. block 128 Run 1 at ~35.7 ms, block 256 Run 1 at ~21.9 ms). These are likely due to cold start (first kernel launch in the process incurs driver/context setup), GPU power-state ramp-up, or system noise on a shared node. Excluding obvious outliers, 128 and 256 give the best FLOPS (~8–9.5e+10).

### Explanations

#### Task 1 — Error-checking macro
All CUDA API calls are wrapped with the `CUDA_CHECK` macro (Week 2 style) so that any failure is reported with file and line and the program exits.

#### Task 2 — Block sizes and grid size
Tested block sizes **32, 64, 128, 256** (threads per block). Grid size (number of blocks) is determined at runtime: `gridSize = (N + BLOCK_SZ - 1) / BLOCK_SZ` with N = 2^30, so e.g. 32 → 33,554,432 blocks, 256 → 4,194,304 blocks.

#### Task 3 — FLOPS for each run
See results above. Typical: 32 → ~3.7–4.0e+10 FLOPS, ~28 ms; 64 → ~6.5–6.9e+10, ~16 ms; 128 → ~7.7–9.5e+10, ~12 ms; 256 → ~7.7–9.3e+10, ~13 ms. Best: 128 and 256 (excluding outliers).

## Problem 2

### Results
Run 1: 
```shell
1D config: grid 262144, block 256, time 2.019 ms, FLOPS 3.325e+10
2D config: grid (256,256), block (32,32), time 0.629 ms, FLOPS 1.067e+11

Comparison: 1D FLOPS = 3.325e+10, 2D FLOPS = 1.067e+11
```

Run 2:
```shell
1D config: grid 262144, block 256, time 1.895 ms, FLOPS 3.541e+10
2D config: grid (256,256), block (32,32), time 0.630 ms, FLOPS 1.066e+11

Comparison: 1D FLOPS = 3.541e+10, 2D FLOPS = 1.066e+11
```

Run 3:
```shell
1D config: grid 262144, block 256, time 1.849 ms, FLOPS 3.629e+10
2D config: grid (256,256), block (32,32), time 0.632 ms, FLOPS 1.062e+11

Comparison: 1D FLOPS = 3.629e+10, 2D FLOPS = 1.062e+11
```

Run 4:
```shell
1D config: grid 262144, block 256, time 1.942 ms, FLOPS 3.456e+10
2D config: grid (256,256), block (32,32), time 0.626 ms, FLOPS 1.073e+11

Comparison: 1D FLOPS = 3.456e+10, 2D FLOPS = 1.073e+11
```

Run 5:
```shell
1D config: grid 262144, block 256, time 1.942 ms, FLOPS 3.456e+10
2D config: grid (256,256), block (32,32), time 0.626 ms, FLOPS 1.073e+11

Comparison: 1D FLOPS = 3.456e+10, 2D FLOPS = 1.073e+11
```

### Explanations

#### Task 1 — Global thread index
* For the 1D configuration, `tid = blockIdx.x * blockDim.x + threadIdx.x`
* For the 2D configuration, there is no single “global linear index” in the same sense; the row and column of the matrix are:
    * `i = blockIdx.y * blockDim.y + threadIdx.y`
    * `j = blockIdx.x * blockDim.x + threadIdx.x`
* The global identity of the thread is the pair `(i,j)`

#### Task 2 — Mapping to matrix element (i, j)
* The matrix is flattened into a row-major linear array for 1D
* In other words, `i = tid / cols` and `j = tid % cols` (cols is 8192 for this problem)
* For the 2D configuration, each thread with `(blockIdx, threadIdx)` is assigned to exactly one element `(i,j)`

#### Task 3 — Kernel time and FLOPS
* 1D ≈ 1.9 ms, ~3.5e+10 FLOPS
* 2D ≈ 0.63 ms, ~1.07e+11 FLOPS

#### Task 4 — Performance comparison
The 2D configuration achieves roughly 3× higher FLOPS than 1D (~1.07e+11 vs ~3.5e+10). With 2D blocks (32×32), threads that share the same row `i` and have consecutive `j` access consecutive memory in row-major A, B, and C, so the GPU can coalesce their accesses into fewer transactions. The 2D grid/block layout also matches the matrix layout, improving cache locality. In 1D, consecutive `tid` does give consecutive indices, but the 2D layout still wins in practice on this kernel.

We look at a simpler example — coalescing on a 1D array (vector addition, N=10, block size 4). We launch 3 blocks. Each thread has `tid = blockIdx.x*4 + threadIdx.x` and does `C[tid] = A[tid] + B[tid]`. In thread order (block 0 then block 1 then block 2, threadIdx.x 0,1,2,3 in each block), the indices used are:

| Block | threadIdx.x | tid | Accesses A[tid], B[tid], C[tid] |
|-------|-------------|-----|----------------------------------|
| 0 | 0 | 0 | A[0], B[0], C[0] |
| 0 | 1 | 1 | A[1], B[1], C[1] |
| 0 | 2 | 2 | A[2], B[2], C[2] |
| 0 | 3 | 3 | A[3], B[3], C[3] |
| 1 | 0 | 4 | A[4], B[4], C[4] |
| 1 | 1 | 5 | A[5], B[5], C[5] |
| … | … | … | … |

So consecutive threads access consecutive addresses (0, 1, 2, 3, 4, 5, …). The GPU can merge these into one or a few contiguous memory transactions — that is coalesced access, which is why 1D vector addition can still be efficient.

For the 2D array, we also do a simple example. We have a 4×4 matrix (row-major; element (i,j) is at index `i*4 + j`). Grid is 2×2 blocks, each block 2×2 threads. So `i = blockIdx.y*2 + threadIdx.y`, `j = blockIdx.x*2 + threadIdx.x`. Each thread does `C[i*4+j] = A[i*4+j] + B[i*4+j]`. In CUDA order, `threadIdx.x` varies fastest. For the first block (blockIdx.x=0, blockIdx.y=0):

| threadIdx.x | threadIdx.y | i | j | idx = i×4+j | Accesses A[idx], B[idx], C[idx] |
|-------------|-------------|---|---|-------------|----------------------------------|
| 0 | 0 | 0 | 0 | 0 | A[0], B[0], C[0] |
| 1 | 0 | 0 | 1 | 1 | A[1], B[1], C[1] |
| 0 | 1 | 1 | 0 | 4 | A[4], B[4], C[4] |
| 1 | 1 | 1 | 1 | 5 | A[5], B[5], C[5] |

So in thread order, the indices used are 0, 1, 4, 5 (not consecutive). The first two threads (same row i=0) access 0, 1 (consecutive); the next two (row i=1) access 4, 5 (consecutive). So within each row of the block we get consecutive accesses, but the block as a whole does not. With larger 32×32 blocks, an entire row of 32 threads has consecutive `j`, so they access 32 consecutive indices — coalesced along that row. The toy 2×2 case shows why small 2D blocks can be non-coalesced; 32×32 recovers coalescing per row.

So why does 2D still win even though a tiny 2D block is non-coalesced? In a toy 2×2 block, the first four threads (in CUDA order) access indices 0, 1, 4, 5 — not consecutive, so that would be non-coalesced. But we use 32×32 blocks. In a 32×32 block, the 32 threads with the same `threadIdx.y` and `threadIdx.x = 0,1,…,31` all share the same row `i` and have consecutive `j`. In row-major layout, that row is stored in consecutive memory, so those 32 threads access consecutive addresses. 

So the fast 2D run is not “non-coalesced”; it gets coalescing too. It beats 1D because the 2D layout matches the matrix (better cache locality, no division/modulo to compute `i`, `j`), and warps align with rows (threadIdx.x is the fast dimension), so the hardware can use memory bandwidth more effectively. The “non-coalesced” pattern only appears in the small 2×2 example; at 32×32, 2D is coalesced and wins on top of that.

---

## Problem 3

### Results

Run 1:
```shell
Block (8 x 8), grid (1024 x 1024), time 906.387 ms, FLOPS 1.213e+12
Block (16 x 16), grid (512 x 512), time 469.795 ms, FLOPS 2.340e+12
Block (32 x 32), grid (256 x 256), time 362.094 ms, FLOPS 3.037e+12
```

Run 2:
```shell
Block (8 x 8), grid (1024 x 1024), time 892.169 ms, FLOPS 1.232e+12
Block (16 x 16), grid (512 x 512), time 468.802 ms, FLOPS 2.345e+12
Block (32 x 32), grid (256 x 256), time 362.063 ms, FLOPS 3.037e+12
```

Run 3:
```shell
Block (8 x 8), grid (1024 x 1024), time 888.083 ms, FLOPS 1.238e+12
Block (16 x 16), grid (512 x 512), time 468.342 ms, FLOPS 2.348e+12
Block (32 x 32), grid (256 x 256), time 362.083 ms, FLOPS 3.037e+12
```

Run 4:
```shell
Block (8 x 8), grid (1024 x 1024), time 891.798 ms, FLOPS 1.233e+12
Block (16 x 16), grid (512 x 512), time 467.115 ms, FLOPS 2.354e+12
Block (32 x 32), grid (256 x 256), time 362.090 ms, FLOPS 3.037e+12
```

Run 5:
```shell
Block (8 x 8), grid (1024 x 1024), time 892.513 ms, FLOPS 1.232e+12
Block (16 x 16), grid (512 x 512), time 468.685 ms, FLOPS 2.346e+12
Block (32 x 32), grid (256 x 256), time 362.081 ms, FLOPS 3.037e+12
```

Summary: 
* 8×8 ≈ 892 ms, ~1.23e+12 FLOPS
* 16×16 ≈ 468 ms, ~2.35e+12 FLOPS
* 32×32 ≈ 362 ms, ~3.04e+12 FLOPS.

### Explanations

#### Task 1 — Thread indices to (i, j)
With a 2D grid and 2D blocks:
- Row of C: `i = blockIdx.y * blockDim.y + threadIdx.y`
- Column of C: `j = blockIdx.x * blockDim.x + threadIdx.x`
Each thread is assigned to exactly one element C(i, j)

#### Task 2 — Inner product for C(i, j)
* Each thread computes C(i,j) = Σ_k A(i,k) * B(k,j) (sum over k = 0 to K−1)
* In code: loop over k, accumulate `A[i*K + k] * B[k*N + j]`, then write `C[i*N + j] = sum`
* So one inner product per thread (~2K FLOPs per element)

#### Task 3 — Kernel time and FLOPS
Measured with CUDA events
FLOPS = 2·M·N·K / time_sec (each C element does K multiplies and K−1 adds)

#### Task 4 — Three 2D block sizes
Tested 8×8, 16×16, and 32×32
Reported FLOPS: 
* 8×8 ~1.23e+12
* 16×16 ~2.35e+12
* 32×32 ~3.04e+12

#### Task 5 — Performance comparison
32×32 is clearly best (~2.5× faster than 8×8). Larger blocks give better occupancy (more warps per SM to hide memory latency) and align with the 2D access pattern for coalescing (e.g. consecutive j → consecutive reads along a row of A). 8×8 has only 64 threads per block, so lower occupancy and more blocks; 16×16 is in between. 32×32 is a common sweet spot for this naive matmul kernel before register/shared-memory limits bite.

**Toy example (performance difference):** Consider a small 16×16 matrix so we can compare block shapes. Each thread computes one C(i,j) by reading row i of A and column j of B.

| Block size | Blocks (grid) | Threads per block | Threads in one row of block | Consecutive A reads per row |
|------------|---------------|-------------------|-----------------------------|------------------------------|
| 8×8        | 2×2 = 4       | 64                | 8                           | 8 (one short coalesced chunk)|
| 16×16      | 1×1 = 1       | 256               | 16                          | 16                           |
| 32×32      | 1×1 = 1       | 1024              | 32                          | 32 (one long coalesced chunk)|

With **8×8 blocks**, each row of the block has only 8 threads with consecutive j, so they issue one coalesced read of 8 elements of A. With **32×32 blocks**, 32 threads share a row and read 32 consecutive elements — a longer coalesced segment, so fewer transactions per row of A. Also, 32×32 has 1024 threads per block (many warps) vs 64 for 8×8, so **occupancy** is much higher: more warps per SM to hide memory latency. So 32×32 wins on both coalescing (longer contiguous reads) and occupancy (more threads per block), matching the observed FLOPS: 8×8 ~1.23e+12, 32×32 ~3.04e+12.
