# Linux profiling (cluster)

PBS scripts for running Nsight Compute profiling on a Linux cluster (e.g. NSCC ASPIRE 2A). Scripts use the project root one level above this folder so you can submit from the Assignment 2 directory:

```bash
cd /path/to/SC4064/Assignment\ 2
qsub profiling_linux/profile_part_a.pbs
qsub profiling_linux/profile_part_b_cusparse.pbs
qsub profiling_linux/profile_part_b_cublas.pbs
```

- **profile_part_a.pbs** – Part A representative case (A1_global, 32×8, L=8).
- **profile_part_b_cusparse.pbs** – Part B cuSPARSE (L=8).
- **profile_part_b_cublas.pbs** – Part B cuBLAS (L=1).

Outputs (plain logs, smoke logs, final logs, and `.ncu-rep` files) are written in the Assignment 2 directory. On clusters that block GPU performance counters (e.g. ERR_NVGPUCTRPERM), use a machine where counters are allowed (e.g. local Linux or Windows with Nsight Compute).
