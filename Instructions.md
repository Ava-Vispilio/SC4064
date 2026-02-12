# SC4064 GPU Programming  
## Assignment 1  

---

# Objectives

The learning objectives of this assignment are:

1. Learn how to compile and run GPU programs on a supercomputer.  
2. Understand how to calculate CUDA thread indices and distribute work across threads.  
3. Explore different grid and block configurations and analyze their impact on performance.  
4. Use CUDA events to measure kernel execution time and compute FLOPS.  

---

# Platform

The CUDA programs in this assignment are designed to run on NVIDIA GPUs. At NSCC Singapore, the ASPIRE 2A supercomputer provides access to NVIDIA A100 GPUs, which we encourage you to use for this assignment. Since this homework focuses on single-GPU programming, you may also use your personal computer or workstation if it has a working CUDA driver and CUDA Toolkit installed. Please document your computing environment if you are not using ASPIRE 2A.

---

# Problem 1. Vector Addition (2 Marks)

Implement a vector addition program using a 1D grid and 1D block configuration. Each vector element should be of type `float`, and the vector length should be at least `2^30`. Initialize the two input vectors with random values in the range `0.0f` to `100.0f`.

## Tasks

1. Wrap all CUDA API calls using the error-checking macro introduced in Week 2’s lecture.  
2. Test the following block sizes (i.e., number of threads per block):  
   - 32  
   - 64  
   - 128  
   - 256  

   For each configuration, your program should determine the grid size (number of blocks) at runtime.  
3. Report the FLOPS achieved for each run.  

---

# Problem 2. Matrix Addition (3 Marks)

Implement a CUDA program that adds Matrix A to Matrix B and stores the result in Matrix C. Each matrix has dimensions **8192 × 8192**, and all elements are of type `float`.

Initialize the matrices as follows:

- **A and B:** random values in the range `[0.0f, 100.0f]`.  
- **C:** all elements initialized to `0.0f`.  

Each CUDA thread should compute the sum of one corresponding element from A and B. You will implement two versions of the CUDA kernel, each using a different grid/block configuration:

- **1D Configuration:** a 1D grid with 1D blocks.  
- **2D Configuration:** a 2D grid with 2D blocks.  

You may choose appropriate grid and block dimensions for each configuration.

## Tasks

1. Show how the global thread index is computed in each configuration.  
2. Explain how the thread index is mapped to a matrix element `(i, j)`.  
3. Measure the kernel execution time and compute the achieved FLOPS for each configuration.  
4. Compare the performance of the two configurations and explain any observed differences.  

---

# Problem 3. Matrix Multiplication (5 Marks)

Implement a CUDA program to compute the matrix multiplication `C = A × B`.

- Matrix A has dimensions `M × K`  
- Matrix B has dimensions `K × N`  
- Matrix C has dimensions `M × N`  

All matrix elements are of type `float`.

For this task, use square matrices with:

```
M = K = N = 8192
```

Initialize the matrices as follows:

- **A and B:** random values in the range `[0.0f, 1.0f]`.  
- **C:** all elements initialized to `0.0f`.  

Each CUDA thread should compute one element of matrix C. Use a 2D grid with 2D blocks for your kernel launch. You may choose appropriate grid and block sizes.

## Tasks

1. Show how the global thread indices (`blockIdx`, `threadIdx`) are used to compute the row and column indices `(i, j)` of matrix C.  
2. Explain how each thread computes the inner product for `C(i, j)`.  
3. Measure the kernel execution time and compute the achieved FLOPS.  
4. Experiment with at least three different 2D block sizes (e.g., `8 × 8`, `16 × 16`, `32 × 32`) and report the achieved FLOPS for each configuration.  
5. Discuss the observed performance differences among the different block sizes.  

---

# Submission

Please compress all required files into a single ZIP archive and submit it via NTULearn. You have two weeks to complete this assignment. The assignment is due by **Friday, Week 6: February 13, 2026, 11:59 PM**. Late submissions will not be accepted.

## 1. Source Code and Job Submission Script

- One CUDA source file (`.cu`) for each problem.  
- Please include sufficient comments to ensure that your code is easy to read and understand.  
- Also include the corresponding job submission scripts and any Makefiles used to build your code.  

## 2. Report

A single report (**≤ 3 pages**) that documents all tasks in this assignment. The report should be clearly structured by problems and include:

### Problem 1: Vector Addition

- Description of the block size configurations tested.  
- Kernel execution time and FLOPS for each block size.  

### Problem 2: Matrix Addition

- Thread index calculations for both the 1D and 2D configurations.  
- Mapping from thread indices to matrix indices `(i, j)`.  
- Kernel execution time and FLOPS for each configuration.  
- Performance comparison and explanation of observed differences.  

### Problem 3: Matrix Multiplication

- Mapping from CUDA thread indices to matrix element indices `(i, j)`.  
- Description of how each thread computes the inner product.  
- Kernel execution time and FLOPS for at least three 2D block sizes.  
- Performance comparison and explanation of observed differences.  
