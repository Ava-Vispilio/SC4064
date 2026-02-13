# Learnings

## Error checking macro

Most files need this to print errors:

```c
// Error Checking Macro (from Week 2)
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (at %s:%d)\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit((int)err); \
    } \
} while(0)
```
Read errors using: `cat $(ls -t *.o* 2>/dev/null | head -1)`

---

## `__global__`

This is a **kernel**: it runs on the GPU and is launched from the host with `vecAdd<<<gridSize, BLOCK_SZ>>>(...)`. It cannot be called like a normal C function from the host.

---

## `__restrict__`

Tells the compiler that `A`, `B`, and `C` do not overlap in memory, so it can optimize loads/stores (e.g. reorder or cache) more aggressively.

---

## How the cudaEvent start/stop sequence works

Order of operations:

| Step | Call | What happens |
|------|------|--------------|
| 1 | `cudaEventCreate(&start)` | Creates a "start" event object (no timestamp yet). |
| 2 | `cudaEventCreate(&stop)` | Creates a "stop" event object. |
| 3 | `cudaEventRecord(start)` | Records "start" on the current GPU stream: "when all previously queued work on this stream has finished, record this time." So this marks the time before the kernel. |
| 4 | `vecAdd<<<...>>>` | Kernel is enqueued on the same stream (runs after start is recorded). |
| 5 | `cudaEventRecord(stop)` | Enqueues "stop": "when all work up to and including the kernel has finished, record this time." So stop is recorded after the kernel completes. |
| 6 | `cudaEventSynchronize(stop)` | Host blocks until the stop event has been recorded. That implies the kernel has finished. So you're waiting for the GPU to finish the kernel (and the stop record). |
| 7 | `cudaEventElapsedTime(&ms, start, stop)` | Computes elapsed time (in ms) between the two recorded timestamps. That's the kernel execution time. |

---

## What is a thread? What is a block?

### Thread

The smallest unit of work on the GPU. Your kernel code runs once per thread. Each thread has:

- **`threadIdx.x`**: its index inside its block (0, 1, 2, …).
- **`blockIdx.x`**: which block it belongs to (0, 1, 2, …).

### Block

A group of threads that run together (same block can share resources and sync). The number of threads in a block is **block size** (your `BLOCK_SZ`). So:

- **`blockDim.x`** = number of threads per block (e.g. 4 or 256).
- **`gridDim.x`** = number of blocks in the grid.

### Grid

The full set of blocks you launch. **Total threads = `gridDim.x` × `blockDim.x`.**

So: one **grid** → many **blocks** → each block has many **threads**. Each thread runs the same kernel code but with different `blockIdx.x` and `threadIdx.x`, so it can do a different part of the work (e.g. a different element index).

---

## The formula: global index = block index × block size + thread index

We want one global index `tid` per thread so that:

1. Every element is handled by exactly one thread, and
2. We can use `tid` as the array index: `C[tid] = A[tid] + B[tid]`.

In 1D:

- **block index** = `blockIdx.x` (which block: 0, 1, 2, …).
- **block size** = `blockDim.x` (threads per block).
- **thread index** (inside block) = `threadIdx.x` (0 to `blockDim.x - 1`).

So:

- All threads in block 0 have `blockIdx.x = 0`. Their global indices are 0, 1, 2, …, `blockDim.x - 1`.
- All threads in block 1 have `blockIdx.x = 1`. Their global indices start right after block 0: `blockDim.x`, `blockDim.x + 1`, …, `2*blockDim.x - 1`.
- And so on.

The unique global index for this thread is:

```text
tid = blockIdx.x × blockDim.x + threadIdx.x
```

That's "block index × block size + thread index (in block)".

---

## Numerical example: vector addition with N = 10, block size = 4

- **Vector length** n = 10 (10 elements).
- **Block size** = 4 threads per block.
- **Grid size** = number of blocks = ceil(10 / 4) = **3 blocks**.

So we launch 3 blocks × 4 threads = 12 threads. Only 10 do useful work; 2 will have `tid >= n` and skip the write.

### How the indices work

**Block 0** (`blockIdx.x = 0`):

| threadIdx.x | tid = 0×4 + threadIdx.x | Does |
|-------------|-------------------------|------|
| 0 | 0 | C[0] = A[0] + B[0] |
| 1 | 1 | C[1] = A[1] + B[1] |
| 2 | 2 | C[2] = A[2] + B[2] |
| 3 | 3 | C[3] = A[3] + B[3] |

**Block 1** (`blockIdx.x = 1`):

| threadIdx.x | tid = 1×4 + threadIdx.x | Does |
|-------------|-------------------------|------|
| 0 | 4 | C[4] = A[4] + B[4] |
| 1 | 5 | C[5] = A[5] + B[5] |
| 2 | 6 | C[6] = A[6] + B[6] |
| 3 | 7 | C[7] = A[7] + B[7] |

**Block 2** (`blockIdx.x = 2`):

| threadIdx.x | tid = 2×4 + threadIdx.x | Does |
|-------------|-------------------------|------|
| 0 | 8 | C[8] = A[8] + B[8] |
| 1 | 9 | C[9] = A[9] + B[9] |
| 2 | 10 | tid ≥ n → no write |
| 3 | 11 | tid ≥ n → no write |

**Summary:**

- **Block index** tells you "which chunk" of the vector (block 0 → indices 0–3, block 1 → 4–7, block 2 → 8–11).
- **Thread index** tells you "which position inside that chunk."
- **Global index** = that chunk's start + position = **block index × block size + thread index**.
