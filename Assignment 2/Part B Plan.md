---
name: Part B CUDA libraries plan
overview: Implement the Part B wave solver as a standalone self-driving CUDA-libraries driver that runs both cuSPARSE and cuBLAS experiments, reports structured timing data, and optionally exports snapshots for later visualization.
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

Snapshot export is implemented but disabled by default through `kExportSnapshots = false`. When enabled, CSV files are written for selected timesteps so plotting can be done later on the host.

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
set export_snapshots = false by default
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
        print SETUP
        run cuSPARSE backend
        print RESULT

if run cuBLAS:
    for each L in cublas_lengths:
        nx = L/dx + 1
        ny = L/dy + 1
        build initial u_prev and u_curr
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

## Visualization Plan That Matches The Code

Visualization support is already built in but turned off by default.

When snapshot export is enabled, the program writes CSV snapshots for:

- step `0`
- step `steps/4`
- step `steps/2`
- step `3*steps/4`
- step `steps`

These CSV files are intended to be processed later by the host-side plotting script `Assignment 2/plot_part_b_snapshots.py` to generate:

- 2D heatmaps
- surface plots
- optional animation

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
- structured `DEVICE`/`CONFIG`/`MEASURE`/`SETUP`/`RESULT` output
- `SKIP` handling for dense cases
- optional CSV snapshot export
- no automatic in-program Part A comparison yet

