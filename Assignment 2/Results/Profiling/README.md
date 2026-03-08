# Profiling results for Assignment 2 report

## Final check: what you need for the report

From the assignment (Instructions.md), your report should use:

| Requirement | Where it comes from |
|-------------|---------------------|
| **Part A – Custom kernel** (A1 global, A2 shared) | `part_a_profile.ncu-rep` + `part_a_profile_plain.log` |
| **Part B – cuSPARSE** | `part_b_cusparse_profile.ncu-rep` + `part_b_cusparse_profile_plain.log` |
| **Part B – cuBLAS** | `part_b_cublas_profile.ncu-rep` + `part_b_cublas_profile_plain.log` |
| **Kernel timing per step** | Plain logs: `kernel_ms_step` / `gpu_ms_step` / `library_ms_step` |
| **Total simulation time** | Plain logs: `sim_ms_total` |
| **Effective memory bandwidth** | Plain logs: `bandwidth_GBps`; formula: 48 bytes per update × updates / kernel time |
| **GPU occupancy and throughput** | Nsight reports: Launch Statistics, occupancy; plain logs: `occupancy_pct`, `active_blocks_per_sm` |
| **Scaling / block size** | Compare runs across domain sizes (L=1,2,4,8) and block configs in plain logs and Nsight |

**In this folder you have:**

- **Part A:** `part_a_profile.ncu-rep`, `part_a_profile_plain.log`
- **Part B cuSPARSE:** `part_b_cusparse_profile.ncu-rep`, `part_b_cusparse_profile_plain.log`
- **Part B cuBLAS:** `part_b_cublas_profile.ncu-rep`, `part_b_cublas_profile_plain.log`

Smoke and final Nsight logs are in `../Debugging Logs/` for reference.

---

## How to read the profiling results

### 1. Plain logs (`.log` in this folder)

- **What they are:** Text output from running the profiling binaries without Nsight (sanity runs).
- **What to use:** Open in any text editor. Each run prints one or more `RESULT` lines with:
  - **Part A:** `kernel_ms_total`, `kernel_ms_step`, `sim_ms_total`, `bandwidth_GBps`, `occupancy_pct`, `active_blocks_per_sm`, `checksum`, `max_error`
  - **Part B:** `gpu_ms_total`, `gpu_ms_step`, `library_ms_total`, `library_ms_step`, `sim_ms_total`, `bandwidth_GBps`, `occupancy_pct`, `checksum`
- **Use for:** Timing, bandwidth, and high-level metrics for your report tables and scaling discussion.

### 2. Nsight Compute reports (`.ncu-rep`)

- **What they are:** Binary report files produced by `ncu --export ...`. They contain detailed per-kernel metrics (launch config, registers, shared memory, occupancy, etc.).
- **How to open:**
  1. Install **NVIDIA Nsight Compute** (you already used the CLI `ncu`; the GUI is usually installed with it).
  2. Start **Nsight Compute** (GUI).
  3. **File → Open** and select the `.ncu-rep` file (e.g. `part_a_profile.ncu-rep`).
- **What you see:**
  - **Summary / report view:** List of kernels (e.g. `wave_step_global`, `update_from_laplacian`, `csr_partition_kernel`, `gemv2N_kernel`). Click a kernel to see its details.
  - **Sections:** e.g. **Launch Statistics** (grid/block size, registers per thread, shared memory, stack size, waves per SM). You can enable more sections (e.g. Memory Workload, Compute Workload) in the GUI for deeper analysis.
  - **Metrics:** Occupancy, throughput, memory bandwidth at the kernel level. Use these to justify “memory-bound vs compute-bound” and block-size choices (Instructions §3.10).
- **Use for:** Kernel implementation quality, occupancy, and critical discussion (e.g. why a kernel is slow, effect of block size, comparison with library kernels).

### 3. Debugging logs (`../Debugging Logs/`)

- **What they are:** Nsight “smoke” (single launch) and “final” (full export) text logs.
- **Use for:** Checking that profiling ran correctly; debugging if a report is missing or a run failed. You do not need to cite these in the report unless you refer to a specific run or error.

---

## Quick reference

- **Bandwidth (assignment formula):** 48 bytes per grid point update; Bandwidth = (total bytes) / (kernel time). The plain logs already compute `bandwidth_GBps` from this.
- **Occupancy:** In Nsight, look at occupancy-related metrics in the kernel details; in plain logs, use `occupancy_pct` and `active_blocks_per_sm`.
- **Scaling:** Use plain logs (and extra runs if you have them) for different domain sizes L=1,2,4,8 and block sizes; use Nsight to explain *why* performance changes (e.g. occupancy, memory throughput).
