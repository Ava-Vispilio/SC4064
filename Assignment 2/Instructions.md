# SC4064 GPU Programming

## Assignment 2: HPC problem and CUDA libraries
## Objectives

The learning objectives of this assignment are:

1. Implement numerical PDE solvers on GPUs
2. Translate mathematical operators into computational forms
3. Apply CUDA libraries appropriately
4. Evaluate memory hierarchy optimizations
5. Visualize scientific computing results
6. Critically compare computational approaches

## Platform

The CUDA programs in this assignment are designed to run on NVIDIA GPUs. At NSCC Singapore, the ASPIRE 2A supercomputer provides access to NVIDIA A100 GPUs, which we encourage you to use for this assignment. Since this homework focuses on single-GPU programming, you may also use your personal computer or workstation if it has a working CUDA driver and CUDA Toolkit installed. Please document your computing environment if you are not using ASPIRE 2A.

# 1 Introduction

In this assignment, you will implement and analyze a GPU-based solver for the two-dimensional wave equation. You are required to:

1. Implement a custom CUDA stencil kernel.
2. Implement an alternative version using CUDA libraries.
3. Visualization.
4. Conduct performance and bandwidth analysis.

The focus of this assignment is not only correctness, but also quantitative performance analysis and critical reasoning.

# 2 Mathematical Background

We consider the 2D wave equation on a square domain:

∂²u/∂t² = c² ∇²u  (1)

where u(t, x, y) is the wave field, c is the wave speed and (x, y) ∈ Ω = [0, 1] × [0, 1], t > 0. Solve the 2D wave equation on a square domain with Dirichlet boundary conditions.

To obtain a well-posed problem, we must specify both initial and boundary conditions.

## 2.1 Initial and Boundary Conditions

It is defined:

• Initial displacement (at t=0):

u(0, x, y) = sin(πx) sin(πy)

• Initial velocity (at t=0):

∂u/∂t (0, x, y) = 0

• Dirichlet boundary condition: u(t,x,y) = 0 for all points on the domain boundary and all times.

## 2.2 Finite Difference Discretization

Use the finite difference scheme provided below.

uⁿ⁺¹ᵢⱼ = 2uⁿᵢⱼ − uⁿ⁻¹ᵢⱼ + (c²Δt² / ΔxΔy) (uⁿᵢ₊₁ⱼ + uⁿᵢ₋₁ⱼ + uⁿᵢⱼ₊₁ + uⁿᵢⱼ₋₁ − 4uⁿᵢⱼ)  (2)

where i,j are spatial indices, and n is the time index. You may assume Δx = Δy. Then

λ² = c²Δt² / ΔxΔy

This is the 5-point stencil.

Parameters for the assignment (you may vary the parameters for resolution study):

• Wave speed: c = 1.0
• Grid spacing: Δx = Δy = 0.01
• Time step: Δt = 0.005

For a standard 2D second-order finite difference scheme, stability requires approximately:

λ ≤ 1/√2

# 3 Implementation Requirements

## 3.1 Part A – Custom CUDA Kernel

You must implement two versions:

(A1) Global Memory Version

• Direct implementation of the stencil.
• No shared memory optimization.

(A2) Shared Memory Tiled Version

• Use shared memory tiling.
• Implement halo loading.
• Handle boundary conditions correctly.

You must:

• Use 2D thread blocks.
• Justify your block size choice.
• Measure kernel execution time using CUDA events.

## 3.2 Part B – CUDA Library Implementation

You must implement at least one of the following:

### Option 1 – cuSPARSE (Recommended)

Reformulate the update as:

uⁿ⁺¹ = 2uⁿ − uⁿ⁻¹ + λ²Luⁿ  (3)

where L is the discrete Laplacian matrix stored in CSR format.

You must:

• Construct the sparse Laplacian matrix.
• Use cusparseSpMV.
• Compare performance with the stencil implementation.

### Option 2 – cuBLAS

Construct a dense matrix L (small grids only) and use:

• cublasDgemv

You must explain why this approach is inefficient.

## 3.3 Wave Field Visualization

Export selected timesteps and generate:

• 2D heatmaps
• Surface plots
• Optional animation

## 3.4 Timing

Measure:

• Kernel time per timestep
• Total simulation time
• Library call time (if applicable)

## 3.5 Effective Memory Bandwidth

For the 5-point stencil:

• 5 reads
• 1 write

In double precision:

6 × 8 = 48 bytes per grid update  (4)

Compute:

Bandwidth = Total bytes transferred / Kernel time  (5)

The purpose of this study is to evaluate GPU performance independently of numerical accuracy effects.

## 3.6 Baseline Configuration

The reference computational domain is:

Ω₀ = [0, 1] × [0, 1]

with uniform grid spacing:

Δx = Δy = constant.

The time step Δt must also remain fixed throughout this study in order to keep the CFL number constant:

λ = cΔt / Δx.

## 3.7 Scaling Strategy

For the performance scaling study:

• Keep Δx, Δy, and Δt fixed.
• Increase the physical domain size.

Define enlarged domains:

Ωₖ = [0, Lₖ] × [0, Lₖ],

where

Lₖ = 1, 2, 4, 8.

Because Δx is fixed, increasing Lₖ increases the number of grid points:

Nₖ = Lₖ / Δx.

Thus, enlarging the domain increases total workload while leaving the numerical scheme unchanged.

## 3.8 Experimental Procedure

For each domain size:

1. Run the solver for a fixed number of time steps.
2. Measure total runtime.
3. Compute effective memory bandwidth.
4. Record GPU occupancy and throughput metrics.

## 3.9 Important Remarks

• This is a performance study, not a resolution study.
• The numerical accuracy does not change because Δx is fixed.
• The CFL condition remains unchanged.
• Only the total computational workload increases.

## 3.10 Analysis Questions

Students must discuss:

• How does runtime scale with total grid points?
• Does performance scale linearly with problem size?
• Is the kernel memory-bound or compute-bound?
• How does block size affect performance?

# 4 Marking Scheme

Component | Weight
--- | ---
Correctness | 20%
Kernel implementation quality | 10%
Library implementation | 10%
Visualization | 20%
Performance analysis | 25%
Critical discussion | 15%

## Submission

Please compress all required files into a single ZIP archive and a pdf report (the report MUST not be compressed in the ZIP archive). 

1. Source Code and Job Submission Script  
One CUDA source file (.cu) includes all different implementations in different functions. Please include sufficient comments to ensure that your code is easy to read and understand. Also include the corresponding job submission scripts and any Makefiles used to build your code.

2. Visualization  
The visualization can be images in jpg or video in mp4.

3. Report  
A single report (<= 3 pages) that documents analysis and reasoning in this assignment. Marks are awarded primarily for analysis and reasoning.

