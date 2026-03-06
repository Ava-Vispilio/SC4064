---
name: Part A CUDA plan
overview: Implement and validate the two required custom CUDA kernels for the 2D wave equation, then benchmark them and justify the chosen 2D block configuration with measurements and hardware-aware reasoning.
todos:
  - id: define-simulation-setup
    content: "Specify the Part A simulation setup: grid size, state arrays, initialization, boundary rules, and timestep loop."
    status: completed
  - id: implement-a1-baseline
    content: Implement the global-memory 5-point stencil kernel and use it as the correctness baseline.
    status: completed
  - id: implement-a2-tiled
    content: Implement the shared-memory tiled kernel with 1-cell halo loading and domain-boundary handling.
    status: completed
  - id: add-timing-and-validation
    content: Plan CUDA-event timing, result validation between kernels, and bandwidth calculation.
    status: completed
  - id: benchmark-block-sizes
    content: Benchmark several 2D block sizes such as 8x8, 16x16, and 32x8, then justify the final choice with performance evidence.
    status: completed
isProject: false
---

# Part A Plan

## Problem Statement And Math

We need to solve the 2D wave equation on a square domain:

`d2u/dt2 = c2 * nabla2(u)`

with:

- Initial displacement: `u(0, x, y) = sin(pi*x) * sin(pi*y)`
- Initial velocity: `du/dt(0, x, y) = 0`
- Dirichlet boundary condition: `u(t, x, y) = 0` on the domain boundary

Using second-order finite differences in time and space, the update at each interior grid point becomes:

`u_next[i,j] = 2*u_curr[i,j] - u_prev[i,j] + lambda2 * (u_curr[i+1,j] + u_curr[i-1,j] + u_curr[i,j+1] + u_curr[i,j-1] - 4*u_curr[i,j])`

where:

- `lambda2 = c*c*dt*dt / (dx*dy)`
- For this assignment: `c = 1.0`, `dx = dy = 0.01`, `dt = 0.005`

With these values:

- `dt2 = 0.005^2 = 2.5e-5`
- `dx*dy = 0.01*0.01 = 1e-4`
- `lambda2 = 2.5e-5 / 1e-4 = 0.25`

The CFL-style stability quantity is:

`lambda = c*dt/dx = 1.0*0.005/0.01 = 0.5`

which satisfies the stated stability guideline `lambda <= 1/sqrt(2) ~= 0.707`.

So the computational task each timestep is simple but very repetitive: every interior point reads its four neighbors and center from the current grid, combines that with the previous grid value, writes one output value, and keeps all boundary cells fixed at zero.

## How We Will Tackle It

We will build Part A in two stages:

- A1: a correct global-memory stencil kernel as the baseline.
- A2: a shared-memory tiled stencil kernel that reduces redundant global loads by reusing data inside each thread block.

The goal is not just to make both versions work. We also need to show why the tiled version should perform better, measure that improvement, and explain whether the kernel is limited mainly by memory traffic rather than arithmetic throughput.

The overall implementation path is:

- Set up the grid, initial condition, and three time-level arrays: `u_prev`, `u_curr`, `u_next`.
- Implement the global-memory kernel first and use it as the correctness reference.
- Implement the shared-memory tiled kernel with a one-cell halo.
- Time both kernels using CUDA events.
- Test multiple 2D block sizes on the A100 and justify the final choice with both hardware reasoning and measured results.

## Implementation Strategy

1. Set fixed simulation parameters from the assignment.

- Use `c = 1.0`, `dx = dy = 0.01`, `dt = 0.005`.
- Precompute `lambda2 = c*c*dt*dt/(dx*dy)`.
- Derive `nx` and `ny` from the domain size and spacing.

1. Create three device arrays.

- `u_prev` for timestep `n-1`
- `u_curr` for timestep `n`
- `u_next` for timestep `n+1`
- Initialize `u_curr(x,y) = sin(pi*x)*sin(pi*y)` and set boundary cells to zero.
- Because initial velocity is zero, initialize `u_prev` consistently with the same starting state or the required first-step treatment chosen for the report.

1. Implement A1: global-memory stencil kernel.

- Launch a 2D grid of 2D thread blocks.
- Each thread maps to one grid point.
- Boundary threads write zero.
- Interior threads read center, left, right, up, down from global memory and write one updated value.

1. Implement A2: shared-memory tiled kernel.

- Allocate shared memory tile sized `(blockDim.y + 2) x (blockDim.x + 2)`.
- Each thread loads its center cell into shared memory.
- Threads at tile edges also load halo cells.
- Synchronize with `__syncthreads()`.
- Interior threads compute the stencil from shared memory, while boundary-domain points still enforce zero.

1. Add timing and performance measurements.

- Use CUDA events around the kernel launch loop.
- Report average kernel time per timestep and total simulation time.
- Compute effective memory bandwidth with the assignment’s 48 bytes per grid update model.

1. Validate before optimizing.

- Compare A1 and A2 outputs after selected timesteps.
- Check that boundary values remain zero.
- Confirm the solution stays numerically stable with the provided `dt`, `dx`, and `dy`.

## Pseudocode

```text
set c, dx, dy, dt
lambda2 = c*c*dt*dt/(dx*dy)
choose domain length L
nx = L/dx + 1
ny = L/dy + 1
allocate host arrays and device arrays: u_prev, u_curr, u_next

for j in 0..ny-1:
    for i in 0..nx-1:
        x = i*dx
        y = j*dy
        if on boundary:
            u_curr[i,j] = 0
        else:
            u_curr[i,j] = sin(pi*x)*sin(pi*y)

initialize u_prev from initial condition / zero-velocity setup
copy u_prev and u_curr to device

create CUDA events
record start event

for step in 0..num_steps-1:
    launch kernel(u_prev, u_curr, u_next, nx, ny, lambda2)
    swap(u_prev, u_curr, u_next)

record stop event
compute elapsed time
copy result back if needed
```

A1 kernel pseudocode:

```text
kernel globalStencil(u_prev, u_curr, u_next, nx, ny, lambda2):
    i = blockIdx.x*blockDim.x + threadIdx.x
    j = blockIdx.y*blockDim.y + threadIdx.y
    if i >= nx or j >= ny: return
    if i == 0 or j == 0 or i == nx-1 or j == ny-1:
        u_next[i,j] = 0
    else:
        center = u_curr[i,j]
        left   = u_curr[i-1,j]
        right  = u_curr[i+1,j]
        down   = u_curr[i,j-1]
        up     = u_curr[i,j+1]
        u_next[i,j] = 2*center - u_prev[i,j] + lambda2*(left + right + up + down - 4*center)
```

A2 kernel pseudocode:

```text
kernel sharedStencil(u_prev, u_curr, u_next, nx, ny, lambda2):
    allocate shared tile[blockDim.y+2][blockDim.x+2]
    i = blockIdx.x*blockDim.x + threadIdx.x
    j = blockIdx.y*blockDim.y + threadIdx.y
    tx = threadIdx.x + 1
    ty = threadIdx.y + 1

    load center cell into tile[ty][tx] if in bounds
    if thread on left tile edge:  load left halo
    if thread on right tile edge: load right halo
    if thread on bottom edge:     load bottom halo
    if thread on top edge:        load top halo
    synchronize

    if i >= nx or j >= ny: return
    if domain boundary:
        u_next[i,j] = 0
    else:
        center = tile[ty][tx]
        left   = tile[ty][tx-1]
        right  = tile[ty][tx+1]
        down   = tile[ty-1][tx]
        up     = tile[ty+1][tx]
        u_next[i,j] = 2*center - u_prev[i,j] + lambda2*(left + right + up + down - 4*center)
```

## Block Size Choice And Justification

We should justify block size in a way that is specific to the A100 rather than treating it as a generic CUDA choice.

### A100-Aware Reasoning

The NVIDIA A100 has:

- Warp size of 32 threads
- Large SM count and high memory bandwidth
- Enough shared memory per SM that our stencil tiles are tiny relative to the hardware limit

For this stencil, arithmetic per grid point is low, so performance is expected to be dominated mostly by memory behavior and execution efficiency. That means the best block size is usually the one that:

- Keeps many warps active
- Preserves good memory access patterns
- Uses shared memory efficiently in the tiled kernel
- Does not create too much halo overhead

### Candidate Block Sizes

We should benchmark at least these 2D candidates:

- `8x8` = 64 threads per block
- `16x16` = 256 threads per block
- `32x8` = 256 threads per block

These are good starting points because:

- They are all valid 2D thread blocks.
- They produce whole numbers of warps.
- `16x16` and `32x8` provide enough work per block to keep the A100 busy.
- They let us compare square and rectangular tiles.

### Halo Overhead Calculation

For the shared-memory version, a `Bx x By` interior tile needs a `(Bx+2) x (By+2)` shared tile.

This gives the following halo overhead ratios:

- `8x8`: load `10x10 = 100` values for `64` outputs, ratio `100/64 = 1.5625`
- `16x16`: load `18x18 = 324` values for `256` outputs, ratio `324/256 = 1.265625`
- `32x8`: load `34x10 = 340` values for `256` outputs, ratio `340/256 = 1.328125`

From this alone, `16x16` is an attractive default because it has lower halo overhead than `8x8` and slightly lower overhead than `32x8`.

### Shared-Memory Footprint

Assuming double precision, each shared tile costs:

- `8x8`: `100 * 8 = 800` bytes
- `16x16`: `324 * 8 = 2592` bytes
- `32x8`: `340 * 8 = 2720` bytes

These are all very small on an A100, so shared-memory capacity is unlikely to be the limiting factor. That means our justification should focus more on halo efficiency, occupancy, and measured runtime than on shared-memory exhaustion.

### Practical Justification Strategy

The plan is:

- Start with `16x16` as the first implementation choice.
- Benchmark `8x8`, `16x16`, and `32x8` on the same domain and timestep count.
- Record kernel time, effective bandwidth, and occupancy/throughput metrics.
- Select the block size with the best or near-best runtime.

If `16x16` wins or is very close to the best, the justification is straightforward:

- 256 threads per block is a strong fit for the A100.
- The tile has favorable halo-to-work ratio.
- Shared-memory cost is low.
- Measured runtime confirms the choice.

If `32x8` wins, we justify it using memory-access behavior and measurements instead of insisting on a square tile. If `8x8` wins, we would need stronger measurement evidence because its lower tile efficiency makes it less compelling on paper.

## Expected Deliverables For Part A

- One correct global-memory kernel.
- One correct shared-memory tiled kernel with halo loading.
- CUDA-event timing results.
- A short comparison of A1 vs A2.
- A short block-size justification backed by benchmark data.

## Results Collection And Report Integration

Because the runs will likely happen on ASPIRE 2A, we should treat result collection as part of the implementation plan, not something to clean up later.

### What We Need To Save From Each Run

For each kernel version and each tested block size, save:

- GPU name and environment details
- Domain size and grid dimensions
- Number of timesteps
- Kernel time per timestep
- Total simulation time
- Effective bandwidth
- Any occupancy or throughput metrics collected from the profiler
- Optional correctness checksum or error metric against the baseline

### Recommended Output Format

Have the program print results in a structured one-line format per experiment so they are easy to copy into a spreadsheet or the report. For example:

```text
kernel=A2_shared block=16x16 nx=101 ny=101 steps=1000 kernel_ms=... total_ms=... bandwidth_GBps=... occupancy=...
```

This makes it easy to:

- Paste raw results from ASPIRE 2A into notes or a spreadsheet
- Group by kernel and block size
- Build final summary tables for the report

### How We Will Compile Results Into The Report

We should organize the analysis section around three compact result views:

1. Correctness summary

- A short statement that A2 matches A1 within expected floating-point tolerance.

1. Performance table

- Columns: kernel version, block size, grid size, average kernel time, total runtime, effective bandwidth

1. Short interpretation paragraph

- Explain which version is faster
- Explain whether the kernel appears memory-bound
- Explain why the chosen block size is justified on the A100

If you copy over raw ASPIRE 2A results later, we can turn them into:

- A clean comparison table
- A short set of bullet conclusions for the report
- A concise narrative for the Part A analysis section

## Notes For The Later Report

The Part A report section should explicitly connect the implementation back to the required analysis:

- Why the shared-memory version reduces redundant global accesses
- Whether the stencil behaves as a memory-bound kernel on the A100
- How runtime changes with problem size
- Why the selected block size is reasonable on the tested GPU

