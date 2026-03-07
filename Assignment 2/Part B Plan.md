---
name: Part B CUDA libraries plan
overview: Implement the Part B wave solver as a standalone self-driving CUDA-libraries driver that runs both cuSPARSE and cuBLAS experiments, reports structured timing data, exports snapshots for one representative visualization run, and provides profiling-only representative cases for final report evidence.
todos:
  - id: refactor-common-driver
    content: Build the standalone Part B driver with shared setup, structured reporting, and fixed experiment lists.
    status: completed
  - id: build-csr-laplacian
    content: Construct the 2D Laplacian in CSR format for each cuSPARSE domain size with explicit Dirichlet-boundary handling.
    status: completed
  - id: add-cusparse-execution
    content: Implement the Part B timestep loop using cusparseSpMV plus a simple update kernel for u_next.
    status: completed
  - id: add-partb-timing-validation
    content: Measure library-call time and total simulation time, and report checksums plus current comparison status.
    status: completed
  - id: run-partb-comparison
    content: Run the fixed-parameter scaling study for cuSPARSE and the limited dense cuBLAS cases, with optional snapshot export for visualization.
    status: completed
isProject: false
---

# Part B Plan

## Goal

Part B is now implemented as its own standalone driver in `Assignment 2/part_b_wave.cu`.

It uses both CUDA-library paths:

- `B1_cusparse` as the main sparse implementation for the full scaling study
- `B2_cublas` as the dense-matrix comparison backend for small domains only

This is the correct description of the current code. The earlier idea of folding Part B into Part A or using only `cuSPARSE` no longer matches the implementation.

## Mathematical Formulation

Both backends use the same reformulation:

`u_next = 2*u_curr - u_prev + lambda2 * (L * u_curr)`

where `L` is the 2D 5-point Laplacian matrix.

Interior rows of `L` contain:

- diagonal `-4`
- left, right, up, down neighbors `+1`

Boundary rows are represented as zero rows in the matrix build, and the final update kernel explicitly forces boundary values back to zero. That matches the current implementation and keeps the Dirichlet condition exact.

As in Part A, the zero-initial-velocity condition is encoded before the loop by constructing:

`u_prev = u_curr - 0.5 * lambda2 * Laplacian(u_curr)`

## What The Implemented Driver Does

The Part B executable is self-driving and does not accept runtime experiment choices.

It currently runs:

- `cuSPARSE` for domain lengths `L = 1, 2, 4, 8`
- `cuBLAS` for domain lengths `L = 1, 2`

The dense `cuBLAS` backend is intentionally restricted because the dense Laplacian becomes too large very quickly. The code also performs a memory-budget check and emits a `SKIP` line when the dense matrix would be unreasonable.

At startup the driver prints:

- `DEVICE`
- `CONFIG`
- `MEASURE`

For each backend/domain case it then prints:

- `SETUP`
- `RESULT`, or `SKIP` for dense cases that do not fit the memory budget

Snapshot export is now enabled, but only for one representative run rather than for every backend/domain case. The code keeps the full performance sweep unchanged and writes CSV snapshots only for `B1_cusparse` at `L = 1.0`.

The driver also now supports compile-time profiling modes used by dedicated PBS scripts:

- `PART_B_PROFILE_MODE=1` profiles representative `B1_cusparse` at `L = 8.0`
- `PART_B_PROFILE_MODE=2` profiles representative `B2_cublas` at `L = 1.0`

In profiling mode, snapshot export is disabled and the PBS profiler scripts use Nsight Compute launch filtering to capture a small representative steady-state launch window rather than the whole run. The code no longer tries to manually gate profiler capture at runtime.

## Backend-Specific Implementation Strategy

### `B1_cusparse`

For the sparse backend, the code:

- builds the Laplacian once in CSR format
- copies `row_offsets`, `col_indices`, and `values` to the device
- creates the cuSPARSE handle and descriptors
- queries the `cusparseSpMV` workspace size
- performs one `cusparseSpMV` per timestep to compute the Laplacian term
- launches a separate CUDA kernel to compute `u_next` and enforce zero boundaries

### `B2_cublas`

For the dense backend, the code:

- constructs the full dense Laplacian in column-major layout
- checks whether the dense matrix fits the configured memory budget
- uses `cublasDgemv` each timestep to compute the Laplacian term
- reuses the same update kernel as the cuSPARSE path

This backend exists mainly for comparison and inefficiency discussion, not for the full scaling study.

## Driver Pseudocode

```text
set constants c, dx, dy, dt, steps
set cusparse_lengths = [1, 2, 4, 8]
set cublas_lengths   = [1, 2]
set export_snapshots = true
set visualization run = B1_cusparse at L = 1.0
set profiling modes:
    mode 1 -> B1_cusparse at L = 8.0
    mode 2 -> B2_cublas at L = 1.0
lambda  = c*dt/dx
lambda2 = c*c*dt*dt/(dx*dy)

print DEVICE
print CONFIG
print MEASURE

if run cuSPARSE:
    for each L in cusparse_lengths:
        nx = L/dx + 1
        ny = L/dy + 1
        build initial u_prev and u_curr
        enable snapshot export only if backend = B1_cusparse and L = 1.0
        print SETUP
        run cuSPARSE backend
        print RESULT

if run cuBLAS:
    for each L in cublas_lengths:
        nx = L/dx + 1
        ny = L/dy + 1
        build initial u_prev and u_curr
        disable snapshot export for all cuBLAS runs
        print SETUP
        run cuBLAS backend
        if dense matrix is too large:
            print SKIP
        else:
            print RESULT
```

## CSR Construction Pseudocode

```text
for each grid point (i, j):
    row = flatten(i, j)

    if point is on the boundary:
        row_offsets[row + 1] = current_nnz
        continue

    append (row, center, -4)
    append (row, left,   +1)
    append (row, right,  +1)
    append (row, down,   +1)
    append (row, up,     +1)
    row_offsets[row + 1] = current_nnz
```

## Per-Timestep Backend Pseudocode

### cuSPARSE Path

```text
record total GPU start event

for step in 0 .. steps-1:
    point x descriptor at d_curr
    point y descriptor at d_laplacian

    record library start
    d_laplacian = cusparseSpMV(L, d_curr)
    record library stop
    accumulate library_ms_total

    launch update kernel:
        if boundary: u_next = 0
        else: u_next = 2*u_curr - u_prev + lambda2*d_laplacian

    swap(d_prev, d_curr, d_next)
    maybe export snapshot for this step

record total GPU stop event
copy final field back
compute checksum and report metrics
```

### cuBLAS Path

```text
check dense matrix size against memory budget
if it does not fit:
    return skipped result

record total GPU start event

for step in 0 .. steps-1:
    record library start
    d_laplacian = cublasDgemv(L_dense, d_curr)
    record library stop
    accumulate library_ms_total

    launch the same update kernel
    swap(d_prev, d_curr, d_next)
    maybe export snapshot for this step

record total GPU stop event
copy final field back
compute checksum and report metrics
```

## Metrics Currently Produced

Each completed Part B `RESULT` line already includes:

- backend label
- domain length and grid size
- timestep count
- total GPU time and GPU time per step
- total library-call time and library-call time per step
- total simulation time
- effective bandwidth using `48 bytes/update`
- occupancy estimate for the update kernel
- active blocks per SM for the update kernel
- checksum
- `max_error`
- note field

Important current detail:

- `max_error` is presently reported as `-1` because Part B does not yet perform an automatic cross-comparison against Part A or between backends inside the executable

So the plan should describe comparison against Part A as a report-analysis step based on logs or later post-processing, not as something the current code already does automatically.

## Profiling Workflow

Part B now supports dedicated profiling-only builds for the final report.

The representative profiling cases are:

- `B1_cusparse` at `L = 8.0`
- `B2_cublas` at `L = 1.0`

These choices are intentional:

- `B1_cusparse`, `L = 8.0` is the scalable library case and the most useful representative workload for throughput analysis
- `B2_cublas`, `L = 1.0` is the small dense case that is still practical to profile while giving direct evidence for inefficiency

The profiling design is:

- keep the normal `1000` timestep count unchanged
- compile a profiling-only binary with `PART_B_PROFILE_MODE`
- run only one representative backend/domain case
- disable snapshot export in profiling mode so visualization I/O does not contaminate profiler results
- use Nsight Compute launch filtering in the PBS scripts to skip early warm-up launches and capture a short steady-state launch window
- do not use runtime `cudaProfilerStart()` / `cudaProfilerStop()` gating in the code path

The profiling scripts are:

- `Assignment 2/profile_part_b_cusparse.pbs`
- `Assignment 2/profile_part_b_cublas.pbs`

The current `ncu` settings are:

- Part A: `--launch-skip 10 --launch-count 1`
- Part B `cuSPARSE`: `--launch-skip 20 --launch-count 10`
- Part B `cuBLAS`: `--launch-skip 20 --launch-count 10`

The longer launch window for Part B is intentional because the library call plus the custom update kernel may involve multiple device launches per timestep, so capturing a short launch range is more reliable than assuming one launch per iteration.

## Visualization Plan That Matches The Code

Visualization support is now enabled for one representative run:

- backend: `B1_cusparse`
- domain length: `L = 1.0`
- timestep count: the full `1000` steps already used by the driver

This is the chosen visualization case because:

- `cuSPARSE` is the main Part B backend and the one that scales across all required domain sizes
- `L = 1.0` is the assignment’s baseline domain, so it is the cleanest representative case to show in the report
- keeping the full `1000` steps preserves consistency with the existing measurements and avoids introducing a special visualization-only timestep count
- exporting only one backend/domain case prevents the run from generating unnecessary CSV files while leaving the rest of the scaling study unchanged

For that representative run, the program writes CSV snapshots for:

- step `0`
- step `steps/4`
- step `steps/2`
- step `3*steps/4`
- step `steps`

These CSV files are intended to be processed later by the host-side plotting script `Assignment 2/plot_part_b_snapshots.py` to generate:

- 2D heatmaps
- surface plots
- optional animation

The code change for this is deliberately minimal: the main experiment loops still run exactly as before, but per-run snapshot export is turned on only when the backend/domain pair matches the selected visualization case.

That means visualization is a two-stage workflow:

1. rerun Part B with snapshot export enabled
2. run Python on the exported CSV files afterward

## Comparison Plan For The Report

The report comparison should now be framed as:

- Part A custom stencil kernels versus Part B `cuSPARSE`
- Part B `cuSPARSE` versus Part B `cuBLAS` on the small domains where dense runs are feasible

The expected discussion remains:

- `cuSPARSE` is the appropriate scalable library backend because the Laplacian is sparse
- `cuBLAS` is mathematically valid but storage-heavy and quickly becomes impractical
- a hand-written stencil may still outperform the sparse-library path because it avoids sparse-index overhead and matrix storage traffic

## What This Plan Now Matches

This updated plan matches the current implementation:

- standalone `part_b_wave.cu`
- both `cuSPARSE` and `cuBLAS` backends
- fixed hardcoded experiment lists
- compile-time profiling modes for representative `cuSPARSE` and `cuBLAS` Nsight Compute runs
- structured `DEVICE`/`CONFIG`/`MEASURE`/`SETUP`/`RESULT` output
- `SKIP` handling for dense cases
- CSV snapshot export enabled only for the representative `B1_cusparse`, `L = 1.0` run
- no automatic in-program Part A comparison yet

