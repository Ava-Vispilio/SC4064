```C
#define CUDA_CHECK(call) d0
{ \
	cudaError_t eer = call; \
	if (err != cudaSuccess) \
		{ \
			fprintf(stderr, "CUDA Error: %s (at %s:%d)\n", \
			cudaGetErrorString(err), __FILE__, __LINE__); \
			exit(err); \
		}
} while(0)
```

Wrap every call that returns an error code

```C
CUDA_CHECK(cudaMalloc(&d_A, size)); // Example
```