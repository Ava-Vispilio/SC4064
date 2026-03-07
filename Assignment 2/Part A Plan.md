---
name: Part A CUDA plan
overview: Implement and validate the two required custom CUDA kernels for the 2D wave equation in a standalone self-driving driver that auto-sweeps the chosen domain sizes and block configurations, then provide a profiling-only representative case for Nsight Compute without disturbing the main benchmark flow.
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

Part A solves the 2D wave equation on a square domain:

`d2u/dt2 = c2 * nabla2(u)`

with:

- initial displacement `u(0, x, y) = sin(pi*x) * sin(pi*y)`
- initial velocity `du/dt(0, x, y) = 0`
- Dirichlet boundary condition `u(t, x, y) = 0` on the domain boundary

Using the assignment discretization, each interior update is:

`u_next[i,j] = 2*u_curr[i,j] - u_prev[i,j] + lambda2 * (u_curr[i+1,j] + u_curr[i-1,j] + u_curr[i,j+1] + u_curr[i,j-1] - 4*u_curr[i,j])`

where:

- `lambda2 = c*c*dt*dt / (dx*dy)`
- `c = 1.0`, `dx = dy = 0.01`, `dt = 0.005`
- therefore `lambda = c*dt/dx = 0.5`, which satisfies the stated stability condition

The actual implementation also encodes the zero-initial-velocity condition directly into `u_prev`, so the same recurrence can be used from timestep 1 onward:

`u_prev = u_curr - 0.5 * lambda2 * Laplacian(u_curr)`

## What The Implemented Driver Does

The implemented Part A program is a standalone self-driving driver in `Assignment 2/part_a_wave.cu`.

It does not take runtime choices from the user. Instead it automatically sweeps:

- domain lengths `L = 1, 2, 4, 8`
- block sizes `8x8`, `16x16`, `32x8`
- both kernels `A1_global` and `A2_shared`

For each experiment, it prints structured lines:

- `DEVICE` once at startup
- `CONFIG` once at startup
- `SETUP` once per domain length
- `RESULT` once per kernel/block/domain combination

This matches the current execution flow used by the PBS script and logs.

The driver now also contains a compile-time profiling mode used by `Assignment 2/profile_part_a.pbs`. In that mode it runs just one representative case:

- kernel: `A1_global`
- block size: `32x8`
- domain length: `L = 8.0`
- steps: still `1000`

The profiling path keeps the normal code structure intact, but narrows execution to a single case so the profiler script can capture a representative steady-state launch window without profiling the full sweep.

## Implementation Strategy

1. Fix the physical and numerical parameters.

- Use `c = 1.0`, `dx = dy = 0.01`, `dt = 0.005`
- Precompute `lambda` and `lambda2`
- Derive `nx` and `ny` from the chosen domain length

1. Build the initial state on the host.

- Allocate `u_prev`, `u_curr`, `u_next`
- Set boundary values to zero
- Fill interior `u_curr` with `sin(pi*x) * sin(pi*y)`
- Back-compute `u_prev` so the first update is consistent with zero initial velocity

1. Implement A1 as the baseline global-memory stencil.

- One thread updates one grid point
- Boundary points are forced to zero
- Interior points read center plus four neighbors directly from global memory

1. Implement A2 as the shared-memory tiled stencil.

- Use a halo-expanded tile of size `(blockDim.y + 2) x (blockDim.x + 2)`
- Load the entire tile cooperatively, not just center cells plus edge-special cases
- This handles partial blocks at domain edges correctly
- After synchronization, compute the same stencil from shared memory

1. Time and validate every configuration.

- Use CUDA events around the full timestep loop
- Record kernel time and host-observed simulation time
- Compute effective bandwidth with the assignment’s `48 bytes/update` model
- Compare `A2_shared` against the corresponding `A1_global` result with `max_error`

## Driver Pseudocode

```text
set constants c, dx, dy, dt, steps
set domain_lengths = [1, 2, 4, 8]
set block_configs = [(8,8), (16,16), (32,8)]
lambda  = c*dt/dx
lambda2 = c*c*dt*dt/(dx*dy)

print DEVICE
print CONFIG

for each domain length L in domain_lengths:
    nx = L/dx + 1
    ny = L/dy + 1

    build host fields u_prev and u_curr
    encode zero initial velocity in u_prev

    print SETUP

    for each block configuration in block_configs:
        run A1_global
        save final field as correctness baseline
        print RESULT

        run A2_shared
        compare final field against A1_global
        print RESULT
```

## Kernel Pseudocode

### A1 Global-Memory Kernel

```text
kernel globalStencil(u_prev, u_curr, u_next, nx, ny, lambda2):
    i = blockIdx.x * blockDim.x + threadIdx.x
    j = blockIdx.y * blockDim.y + threadIdx.y

    if i >= nx or j >= ny:
        return

    if i == 0 or j == 0 or i == nx-1 or j == ny-1:
        u_next[i,j] = 0
        return

    center = u_curr[i,j]
    left   = u_curr[i-1,j]
    right  = u_curr[i+1,j]
    down   = u_curr[i,j-1]
    up     = u_curr[i,j+1]

    u_next[i,j] = 2*center - u_prev[i,j] + lambda2*(left + right + up + down - 4*center)
```

### A2 Shared-Memory Kernel

```text
kernel sharedStencil(u_prev, u_curr, u_next, nx, ny, lambda2):
    allocate shared tile[(blockDim.y + 2) * (blockDim.x + 2)]

    global_i = blockIdx.x * blockDim.x + threadIdx.x
    global_j = blockIdx.y * blockDim.y + threadIdx.y
    tile_pitch  = blockDim.x + 2
    tile_height = blockDim.y + 2
    local_i = threadIdx.x + 1
    local_j = threadIdx.y + 1

    for tile_j from threadIdx.y to tile_height-1 step blockDim.y:
        source_j = blockIdx.y * blockDim.y + tile_j - 1
        for tile_i from threadIdx.x to tile_pitch-1 step blockDim.x:
            source_i = blockIdx.x * blockDim.x + tile_i - 1
            if source_i and source_j are inside the domain:
                tile[tile_j, tile_i] = u_curr[source_j, source_i]
            else:
                tile[tile_j, tile_i] = 0

    synchronize

    if global_i >= nx or global_j >= ny:
        return

    if global_i == 0 or global_j == 0 or global_i == nx-1 or global_j == ny-1:
        u_next[global_i, global_j] = 0
        return

    center = tile[local_j,     local_i]
    left   = tile[local_j,     local_i - 1]
    right  = tile[local_j,     local_i + 1]
    down   = tile[local_j - 1, local_i]
    up     = tile[local_j + 1, local_i]

    u_next[global_i, global_j] =
        2*center - u_prev[global_i, global_j] +
        lambda2*(left + right + up + down - 4*center)
```

This cooperative loading scheme is the important correction relative to the earlier edge-thread-only sketch. It matches the implemented kernel and is robust for partial edge blocks.

## Block Size Choice And Justification

The A100-aware benchmarking plan remains:

- `8x8` for a smaller 64-thread block
- `16x16` for a square 256-thread block
- `32x8` for a rectangular 256-thread block

These are still the right candidates because:

- they produce whole numbers of warps
- `16x16` and `32x8` give 256 threads per block, which is a strong occupancy target on A100-class hardware
- they let us compare square and rectangular tiles
- the shared-memory footprint is tiny in all cases

Halo overhead for the tiled kernel is still:

- `8x8`: `10x10 / 64 = 1.5625`
- `16x16`: `18x18 / 256 = 1.265625`
- `32x8`: `34x10 / 256 = 1.328125`

So the reasoning is unchanged: `16x16` is the best paper choice, but the final justification should come from the measured `RESULT` lines in `Part_A.log`.

## Metrics And Output To Use In The Report

Each `RESULT` line already provides the core Part A report inputs:

- kernel label
- block size
- domain length and grid size
- total kernel time
- kernel time per timestep
- total simulation time
- effective bandwidth
- occupancy estimate
- active blocks per SM
- checksum
- `max_error` against the A1 baseline

That means the remaining Part A write-up can be built directly from the current log without changing the execution flow again.

## Profiling Workflow

Part A now supports a profiling-only build for the final report evidence.

The chosen representative case is:

- `A1_global`
- block `32x8`
- domain length `L = 8.0`

This matches the best custom-kernel configuration seen in the current results and gives a large enough problem to make profiler throughput numbers meaningful.

The profiling design is:

- keep the normal `1000` timestep count unchanged
- disable the usual multi-case sweep by compiling with `PART_A_PROFILE_MODE=1`
- run a single representative case
- let Nsight Compute skip the first `10` kernel launches and capture the next `1` launch
- do not use runtime `cudaProfilerStart()` / `cudaProfilerStop()` gating in the code path

This is implemented through the dedicated PBS script `Assignment 2/profile_part_a.pbs`, which now performs the profiling run in checkpoints:

- enable shell tracing with `set -euxo pipefail`
- verify `ncu` availability with `which ncu` and `ncu --version`
- compile the profiling binary
- run the binary once without `ncu` and save `part_a_profile_plain.log`
- run a lightweight `ncu` smoke test with `--launch-count 1` and save `part_a_profile_ncu_smoke.log`
- finally export the report with `--launch-skip 10 --launch-count 1`

This staged workflow makes it easier to determine whether a failure comes from compilation, the profiling-mode binary itself, the `ncu` environment, or the final filtered profiling command.

## What This Plan Now Matches

This updated plan matches the current implementation:

- standalone `part_a_wave.cu`
- automatic sweep over selected domain lengths and block sizes
- compile-time profiling mode for one representative Nsight Compute run
- A1 used as the per-configuration correctness baseline
- A2 implemented with cooperative halo loading
- structured logging intended for later report tables

