#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (at %s:%d)\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
        exit(static_cast<int>(err__)); \
    } \
} while (0)

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kWaveSpeed = 1.0;
constexpr double kDx = 0.01;
constexpr double kDy = 0.01;
constexpr double kDt = 0.005;
constexpr double kBytesPerUpdate = 48.0;
constexpr int kDefaultSteps = 1000;
constexpr bool kRunGlobal = true;
constexpr bool kRunShared = true;
constexpr bool kBenchmarkBlocks = true;
constexpr double kDomainLengths[] = {1.0, 2.0, 4.0, 8.0};

enum class KernelKind {
    kGlobal,
    kShared,
};

struct Options {
    double domain_length = 1.0;
    int steps = 1000;
    int block_x = 16;
    int block_y = 16;
    bool benchmark_blocks = true;
    bool run_global = true;
    bool run_shared = true;
};

struct RunResult {
    std::string kernel_label;
    int block_x = 0;
    int block_y = 0;
    int nx = 0;
    int ny = 0;
    int steps = 0;
    float kernel_ms_total = 0.0f;
    float kernel_ms_per_step = 0.0f;
    double sim_ms_total = 0.0;
    double bandwidth_gbps = 0.0;
    double occupancy_pct = 0.0;
    int active_blocks_per_sm = 0;
    double checksum = 0.0;
    double max_error = 0.0;
    std::vector<double> final_field;
};

Options default_options() {
    Options options;
    options.steps = kDefaultSteps;
    options.run_global = kRunGlobal;
    options.run_shared = kRunShared;
    options.benchmark_blocks = kBenchmarkBlocks;
    if (options.steps <= 0) {
        fprintf(stderr, "Number of steps must be positive.\n");
        std::exit(EXIT_FAILURE);
    }
    if (!options.run_global && !options.run_shared) {
        fprintf(stderr, "At least one kernel must be enabled.\n");
        std::exit(EXIT_FAILURE);
    }

    return options;
}

int grid_points_from_length(double length, double spacing) {
    const double cells = length / spacing;
    const long long rounded_cells = std::llround(cells);
    if (std::fabs(cells - static_cast<double>(rounded_cells)) > 1e-9) {
        fprintf(stderr, "Domain length %.6f is not an integer multiple of spacing %.6f.\n", length, spacing);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(rounded_cells) + 1;
}

size_t index_2d(int i, int j, int nx) {
    return static_cast<size_t>(j) * static_cast<size_t>(nx) + static_cast<size_t>(i);
}

void initialize_fields(std::vector<double> *u_prev,
                       std::vector<double> *u_curr,
                       int nx,
                       int ny,
                       double dx,
                       double dy,
                       double lambda2) {
    const size_t total_points = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    u_prev->assign(total_points, 0.0);
    u_curr->assign(total_points, 0.0);

    for (int j = 0; j < ny; ++j) {
        const double y = static_cast<double>(j) * dy;
        for (int i = 0; i < nx; ++i) {
            const double x = static_cast<double>(i) * dx;
            const bool is_boundary = (i == 0 || j == 0 || i == nx - 1 || j == ny - 1);
            const size_t idx = index_2d(i, j, nx);
            if (is_boundary) {
                (*u_curr)[idx] = 0.0;
            } else {
                (*u_curr)[idx] = std::sin(kPi * x) * std::sin(kPi * y);
            }
        }
    }

    // Encode the zero initial velocity in u_prev so the main recurrence can be
    // used from the very first timestep: u^1 = u^0 + 0.5 * lambda2 * Laplacian(u^0).
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const size_t center_idx = index_2d(i, j, nx);
            const double center = (*u_curr)[center_idx];
            const double left = (*u_curr)[index_2d(i - 1, j, nx)];
            const double right = (*u_curr)[index_2d(i + 1, j, nx)];
            const double down = (*u_curr)[index_2d(i, j - 1, nx)];
            const double up = (*u_curr)[index_2d(i, j + 1, nx)];
            const double laplacian = left + right + up + down - 4.0 * center;
            (*u_prev)[center_idx] = center - 0.5 * lambda2 * laplacian;
        }
    }
}

double compute_checksum(const std::vector<double> &field) {
    double checksum = 0.0;
    for (double value : field) {
        checksum += value;
    }
    return checksum;
}

double compute_max_abs_diff(const std::vector<double> &lhs, const std::vector<double> &rhs) {
    if (lhs.size() != rhs.size()) {
        return INFINITY;
    }

    double max_diff = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
    }
    return max_diff;
}

__global__ void wave_step_global(const double *__restrict__ u_prev,
                                 const double *__restrict__ u_curr,
                                 double *__restrict__ u_next,
                                 int nx,
                                 int ny,
                                 double lambda2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) {
        return;
    }

    const size_t idx = static_cast<size_t>(j) * static_cast<size_t>(nx) + static_cast<size_t>(i);
    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        u_next[idx] = 0.0;
        return;
    }

    const double center = u_curr[idx];
    const double left = u_curr[idx - 1];
    const double right = u_curr[idx + 1];
    const double down = u_curr[idx - static_cast<size_t>(nx)];
    const double up = u_curr[idx + static_cast<size_t>(nx)];
    u_next[idx] = 2.0 * center - u_prev[idx] + lambda2 * (left + right + up + down - 4.0 * center);
}

__global__ void wave_step_shared(const double *__restrict__ u_prev,
                                 const double *__restrict__ u_curr,
                                 double *__restrict__ u_next,
                                 int nx,
                                 int ny,
                                 double lambda2) {
    extern __shared__ double tile[];

    const int global_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_j = blockIdx.y * blockDim.y + threadIdx.y;
    const int tile_pitch = blockDim.x + 2;
    const int tile_height = blockDim.y + 2;
    const int local_i = threadIdx.x + 1;
    const int local_j = threadIdx.y + 1;
    const bool in_bounds = (global_i < nx && global_j < ny);

    // Cooperative loading covers the full halo-expanded tile, including partial
    // blocks at the domain edges where the last valid point is not the last thread.
    for (int tile_j = threadIdx.y; tile_j < tile_height; tile_j += blockDim.y) {
        const int source_j = blockIdx.y * blockDim.y + tile_j - 1;
        for (int tile_i = threadIdx.x; tile_i < tile_pitch; tile_i += blockDim.x) {
            const int source_i = blockIdx.x * blockDim.x + tile_i - 1;
            double value = 0.0;
            if (source_i >= 0 && source_i < nx && source_j >= 0 && source_j < ny) {
                value = u_curr[static_cast<size_t>(source_j) * static_cast<size_t>(nx) + static_cast<size_t>(source_i)];
            }
            tile[tile_j * tile_pitch + tile_i] = value;
        }
    }

    __syncthreads();

    if (!in_bounds) {
        return;
    }

    const size_t idx = static_cast<size_t>(global_j) * static_cast<size_t>(nx) + static_cast<size_t>(global_i);
    if (global_i == 0 || global_j == 0 || global_i == nx - 1 || global_j == ny - 1) {
        u_next[idx] = 0.0;
        return;
    }

    const double center = tile[local_j * tile_pitch + local_i];
    const double left = tile[local_j * tile_pitch + (local_i - 1)];
    const double right = tile[local_j * tile_pitch + (local_i + 1)];
    const double down = tile[(local_j - 1) * tile_pitch + local_i];
    const double up = tile[(local_j + 1) * tile_pitch + local_i];
    u_next[idx] = 2.0 * center - u_prev[idx] + lambda2 * (left + right + up + down - 4.0 * center);
}

void launch_kernel(KernelKind kernel_kind,
                   const dim3 &grid,
                   const dim3 &block,
                   size_t shared_mem_bytes,
                   const double *d_prev,
                   const double *d_curr,
                   double *d_next,
                   int nx,
                   int ny,
                   double lambda2) {
    if (kernel_kind == KernelKind::kGlobal) {
        wave_step_global<<<grid, block>>>(d_prev, d_curr, d_next, nx, ny, lambda2);
    } else {
        wave_step_shared<<<grid, block, shared_mem_bytes>>>(d_prev, d_curr, d_next, nx, ny, lambda2);
    }
}

RunResult run_configuration(KernelKind kernel_kind,
                            const dim3 &block,
                            int nx,
                            int ny,
                            int steps,
                            double lambda2,
                            const std::vector<double> &initial_prev,
                            const std::vector<double> &initial_curr,
                            const cudaDeviceProp &device_prop) {
    RunResult result;
    result.kernel_label = (kernel_kind == KernelKind::kGlobal) ? "A1_global" : "A2_shared";
    result.block_x = static_cast<int>(block.x);
    result.block_y = static_cast<int>(block.y);
    result.nx = nx;
    result.ny = ny;
    result.steps = steps;

    const size_t total_points = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t bytes = total_points * sizeof(double);
    const size_t shared_mem_bytes =
        (kernel_kind == KernelKind::kShared) ? (block.x + 2) * (block.y + 2) * sizeof(double) : 0;
    auto sim_start = std::chrono::steady_clock::now();

    double *d_prev = nullptr;
    double *d_curr = nullptr;
    double *d_next = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_prev), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_curr), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_next), bytes));
    CUDA_CHECK(cudaMemcpy(d_prev, initial_prev.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_curr, initial_curr.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_next, 0, bytes));

    const dim3 grid((static_cast<unsigned int>(nx) + block.x - 1) / block.x,
                    (static_cast<unsigned int>(ny) + block.y - 1) / block.y);

    int active_blocks_per_sm = 0;
    if (kernel_kind == KernelKind::kGlobal) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm, wave_step_global, static_cast<int>(block.x * block.y), 0));
    } else {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm, wave_step_shared, static_cast<int>(block.x * block.y), shared_mem_bytes));
    }
    const int max_warps_per_sm = device_prop.maxThreadsPerMultiProcessor / device_prop.warpSize;
    const int active_warps = active_blocks_per_sm * static_cast<int>((block.x * block.y + device_prop.warpSize - 1) / device_prop.warpSize);
    result.active_blocks_per_sm = active_blocks_per_sm;
    result.occupancy_pct = std::min(
        100.0, 100.0 * static_cast<double>(active_warps) / static_cast<double>(max_warps_per_sm));

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    CUDA_CHECK(cudaEventRecord(start_event));
    for (int step = 0; step < steps; ++step) {
        launch_kernel(kernel_kind, grid, block, shared_mem_bytes, d_prev, d_curr, d_next, nx, ny, lambda2);
        double *tmp = d_prev;
        d_prev = d_curr;
        d_curr = d_next;
        d_next = tmp;
    }
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&result.kernel_ms_total, start_event, stop_event));
    result.kernel_ms_per_step = result.kernel_ms_total / static_cast<float>(steps);

    result.final_field.assign(total_points, 0.0);
    CUDA_CHECK(cudaMemcpy(result.final_field.data(), d_curr, bytes, cudaMemcpyDeviceToHost));
    auto sim_stop = std::chrono::steady_clock::now();
    result.sim_ms_total = std::chrono::duration<double, std::milli>(sim_stop - sim_start).count();
    result.checksum = compute_checksum(result.final_field);

    const double interior_updates = static_cast<double>(std::max(nx - 2, 0)) * static_cast<double>(std::max(ny - 2, 0));
    const double total_bytes = interior_updates * static_cast<double>(steps) * kBytesPerUpdate;
    result.bandwidth_gbps = total_bytes / (static_cast<double>(result.kernel_ms_total) * 1.0e6);

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(d_prev));
    CUDA_CHECK(cudaFree(d_curr));
    CUDA_CHECK(cudaFree(d_next));

    return result;
}

void print_result_line(const RunResult &result, double domain_length) {
    printf(
        "RESULT kernel=%s block=%dx%d length=%.2f nx=%d ny=%d steps=%d "
        "kernel_ms_total=%.6f kernel_ms_step=%.6f sim_ms_total=%.6f "
        "bandwidth_GBps=%.6f occupancy_pct=%.2f active_blocks_per_sm=%d "
        "checksum=%.12e max_error=%.12e\n",
        result.kernel_label.c_str(),
        result.block_x,
        result.block_y,
        domain_length,
        result.nx,
        result.ny,
        result.steps,
        result.kernel_ms_total,
        result.kernel_ms_per_step,
        result.sim_ms_total,
        result.bandwidth_gbps,
        result.occupancy_pct,
        result.active_blocks_per_sm,
        result.checksum,
        result.max_error);
}

}  // namespace

int main(void) {
    const Options options = default_options();

    CUDA_CHECK(cudaSetDevice(0));

    int device = 0;
    cudaDeviceProp device_prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device));

    const double lambda = kWaveSpeed * kDt / kDx;
    const double lambda2 = (kWaveSpeed * kWaveSpeed * kDt * kDt) / (kDx * kDy);

    printf("DEVICE name=\"%s\" sm_count=%d warp_size=%d max_threads_per_sm=%d shared_mem_per_sm=%zu\n",
           device_prop.name,
           device_prop.multiProcessorCount,
           device_prop.warpSize,
           device_prop.maxThreadsPerMultiProcessor,
           device_prop.sharedMemPerMultiprocessor);
    printf("CONFIG steps=%d run_global=%d run_shared=%d benchmark_blocks=%d lengths=1.00,2.00,4.00,8.00 blocks=8x8,16x16,32x8\n",
           options.steps,
           options.run_global ? 1 : 0,
           options.run_shared ? 1 : 0,
           options.benchmark_blocks ? 1 : 0);

    std::vector<dim3> block_configs;
    if (options.benchmark_blocks) {
        block_configs.push_back(dim3(8, 8));
        block_configs.push_back(dim3(16, 16));
        block_configs.push_back(dim3(32, 8));
    } else {
        block_configs.push_back(dim3(static_cast<unsigned int>(options.block_x),
                                     static_cast<unsigned int>(options.block_y)));
    }

    for (const dim3 &block : block_configs) {
        const unsigned int threads_per_block = block.x * block.y;
        if (threads_per_block == 0 || threads_per_block > static_cast<unsigned int>(device_prop.maxThreadsPerBlock)) {
            fprintf(stderr, "Invalid block configuration %ux%u for this GPU.\n", block.x, block.y);
            return EXIT_FAILURE;
        }
    }

    for (double domain_length : kDomainLengths) {
        const int nx = grid_points_from_length(domain_length, kDx);
        const int ny = grid_points_from_length(domain_length, kDy);
        std::vector<double> initial_prev;
        std::vector<double> initial_curr;
        initialize_fields(&initial_prev, &initial_curr, nx, ny, kDx, kDy, lambda2);

        printf("SETUP length=%.2f dx=%.5f dy=%.5f dt=%.5f c=%.2f lambda=%.6f lambda2=%.6f nx=%d ny=%d steps=%d\n",
               domain_length,
               kDx,
               kDy,
               kDt,
               kWaveSpeed,
               lambda,
               lambda2,
               nx,
               ny,
               options.steps);

        for (const dim3 &block : block_configs) {
            RunResult global_result;
            bool have_global_baseline = false;

            if (options.run_global) {
                global_result = run_configuration(
                    KernelKind::kGlobal, block, nx, ny, options.steps, lambda2, initial_prev, initial_curr, device_prop);
                global_result.max_error = 0.0;
                print_result_line(global_result, domain_length);
                have_global_baseline = true;
            }

            if (options.run_shared) {
                RunResult shared_result = run_configuration(
                    KernelKind::kShared, block, nx, ny, options.steps, lambda2, initial_prev, initial_curr, device_prop);
                if (have_global_baseline) {
                    shared_result.max_error = compute_max_abs_diff(shared_result.final_field, global_result.final_field);
                }
                print_result_line(shared_result, domain_length);
            }
        }
    }

    return 0;
}
