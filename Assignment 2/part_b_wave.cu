#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (at %s:%d)\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
        exit(static_cast<int>(err__)); \
    } \
} while (0)

#define CUSPARSE_CHECK(call) do { \
    cusparseStatus_t status__ = (call); \
    if (status__ != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE Error: %d (at %s:%d)\n", static_cast<int>(status__), __FILE__, __LINE__); \
        exit(static_cast<int>(status__)); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status__ = (call); \
    if (status__ != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error: %d (at %s:%d)\n", static_cast<int>(status__), __FILE__, __LINE__); \
        exit(static_cast<int>(status__)); \
    } \
} while (0)

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kWaveSpeed = 1.0;
constexpr double kDx = 0.01;
constexpr double kDy = 0.01;
constexpr double kDt = 0.005;
constexpr double kBytesPerUpdate = 48.0;
constexpr int kUpdateBlockX = 32;
constexpr int kUpdateBlockY = 8;
constexpr int kDefaultSteps = 1000;
constexpr bool kRunCuSparse = true;
constexpr bool kRunCuBlas = true;
constexpr bool kExportSnapshots = false;
constexpr double kDenseMaxGB = 20.0;
constexpr double kCuSparseLengths[] = {1.0, 2.0, 4.0, 8.0};
constexpr double kCuBlasLengths[] = {1.0, 2.0};
const char kSnapshotDir[] = "part_b_snapshots";

struct Options {
    double domain_length = 1.0;
    int steps = kDefaultSteps;
    bool export_snapshots = kExportSnapshots;
    std::string snapshot_dir = kSnapshotDir;
    double dense_max_gb = kDenseMaxGB;
};

struct RunResult {
    std::string backend_label;
    int nx = 0;
    int ny = 0;
    int steps = 0;
    float gpu_ms_total = 0.0f;
    float gpu_ms_per_step = 0.0f;
    float library_ms_total = 0.0f;
    float library_ms_per_step = 0.0f;
    double sim_ms_total = 0.0;
    double bandwidth_gbps = 0.0;
    double update_occupancy_pct = -1.0;
    int active_blocks_per_sm = -1;
    double checksum = 0.0;
    double max_error = -1.0;
    bool completed = false;
    std::string note;
    std::vector<double> final_field;
};

struct CsrMatrixHost {
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    std::vector<double> values;
};

struct DenseMatrixHost {
    std::vector<double> values;
    size_t bytes = 0;
};

Options default_options() {
    Options options;
    if (options.steps <= 0) {
        fprintf(stderr, "Number of steps must be positive.\n");
        std::exit(EXIT_FAILURE);
    }
    if (options.dense_max_gb <= 0.0) {
        fprintf(stderr, "Dense matrix size limit must be positive.\n");
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

__host__ __device__ inline size_t index_2d(int i, int j, int nx) {
    return static_cast<size_t>(j) * static_cast<size_t>(nx) + static_cast<size_t>(i);
}

std::string format_length_token(double length) {
    char raw[32];
    std::snprintf(raw, sizeof(raw), "%.2f", length);
    std::string token(raw);
    std::replace(token.begin(), token.end(), '.', 'p');
    return token;
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

[[maybe_unused]] double compute_max_abs_diff(const std::vector<double> &lhs, const std::vector<double> &rhs) {
    if (lhs.size() != rhs.size()) {
        return std::numeric_limits<double>::infinity();
    }

    double max_diff = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
    }
    return max_diff;
}

void ensure_directory_exists(const std::string &path) {
    if (path.empty()) {
        return;
    }

    std::string partial;
    partial.reserve(path.size());
    for (size_t i = 0; i < path.size(); ++i) {
        const char ch = path[i];
        partial.push_back(ch);

        const bool at_separator = (ch == '/');
        const bool at_end = (i + 1 == path.size());
        if (!at_separator && !at_end) {
            continue;
        }

        while (partial.size() > 1 && partial.back() == '/') {
            partial.pop_back();
        }
        if (partial.empty()) {
            continue;
        }

        if (mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST) {
            fprintf(stderr, "Failed to create directory '%s': %s\n", partial.c_str(), std::strerror(errno));
            std::exit(EXIT_FAILURE);
        }
    }
}

std::vector<int> build_snapshot_steps(int steps) {
    std::vector<int> selected{0, steps / 4, steps / 2, (3 * steps) / 4, steps};
    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    return selected;
}

void write_snapshot_csv(const std::string &snapshot_dir,
                        const std::string &backend_label,
                        double domain_length,
                        int nx,
                        int ny,
                        int step,
                        const std::vector<double> &field) {
    ensure_directory_exists(snapshot_dir);
    const std::string filename = snapshot_dir + "/" + backend_label + "_L" + format_length_token(domain_length) +
                                 "_step" + std::to_string(step) + ".csv";
    std::ofstream out(filename);
    out << "# backend=" << backend_label
        << ", length=" << domain_length
        << ", step=" << step
        << ", nx=" << nx
        << ", ny=" << ny << '\n';
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (i > 0) {
                out << ',';
            }
            out << field[index_2d(i, j, nx)];
        }
        out << '\n';
    }
}

CsrMatrixHost build_laplacian_csr(int nx, int ny) {
    CsrMatrixHost csr;
    const int total_points = nx * ny;
    csr.row_offsets.resize(static_cast<size_t>(total_points) + 1, 0);
    csr.col_indices.reserve(static_cast<size_t>(std::max(nx - 2, 0)) * static_cast<size_t>(std::max(ny - 2, 0)) * 5);
    csr.values.reserve(csr.col_indices.capacity());

    int nnz = 0;
    csr.row_offsets[0] = 0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int row = static_cast<int>(index_2d(i, j, nx));
            const bool is_boundary = (i == 0 || j == 0 || i == nx - 1 || j == ny - 1);
            if (!is_boundary) {
                csr.col_indices.push_back(static_cast<int>(index_2d(i, j, nx)));
                csr.values.push_back(-4.0);
                csr.col_indices.push_back(static_cast<int>(index_2d(i - 1, j, nx)));
                csr.values.push_back(1.0);
                csr.col_indices.push_back(static_cast<int>(index_2d(i + 1, j, nx)));
                csr.values.push_back(1.0);
                csr.col_indices.push_back(static_cast<int>(index_2d(i, j - 1, nx)));
                csr.values.push_back(1.0);
                csr.col_indices.push_back(static_cast<int>(index_2d(i, j + 1, nx)));
                csr.values.push_back(1.0);
                nnz += 5;
            }
            csr.row_offsets[static_cast<size_t>(row) + 1] = nnz;
        }
    }

    return csr;
}

bool estimate_dense_matrix_bytes(size_t total_points, size_t *dense_bytes) {
    if (total_points == 0 || total_points > std::numeric_limits<size_t>::max() / total_points) {
        return false;
    }
    const size_t entries = total_points * total_points;
    if (entries > std::numeric_limits<size_t>::max() / sizeof(double)) {
        return false;
    }
    *dense_bytes = entries * sizeof(double);
    return true;
}

bool dense_matrix_fits_budget(size_t dense_bytes, double dense_max_gb, size_t free_mem_bytes) {
    const double dense_limit_bytes = dense_max_gb * 1024.0 * 1024.0 * 1024.0;
    const double memory_budget = 0.80 * static_cast<double>(free_mem_bytes);
    return static_cast<double>(dense_bytes) <= dense_limit_bytes &&
           static_cast<double>(dense_bytes) <= memory_budget;
}

DenseMatrixHost build_laplacian_dense_column_major(int nx, int ny) {
    DenseMatrixHost dense;
    const size_t total_points = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    dense.bytes = total_points * total_points * sizeof(double);
    dense.values.assign(total_points * total_points, 0.0);

    auto set_value = [&](size_t row, size_t col, double value) {
        dense.values[col * total_points + row] = value;
    };

    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const size_t row = index_2d(i, j, nx);
            set_value(row, index_2d(i, j, nx), -4.0);
            set_value(row, index_2d(i - 1, j, nx), 1.0);
            set_value(row, index_2d(i + 1, j, nx), 1.0);
            set_value(row, index_2d(i, j - 1, nx), 1.0);
            set_value(row, index_2d(i, j + 1, nx), 1.0);
        }
    }

    return dense;
}

__global__ void update_from_laplacian(const double *__restrict__ u_prev,
                                      const double *__restrict__ u_curr,
                                      const double *__restrict__ laplacian,
                                      double *__restrict__ u_next,
                                      int nx,
                                      int ny,
                                      double lambda2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) {
        return;
    }

    const size_t idx = index_2d(i, j, nx);
    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        u_next[idx] = 0.0;
        return;
    }
    u_next[idx] = 2.0 * u_curr[idx] - u_prev[idx] + lambda2 * laplacian[idx];
}

void compute_update_kernel_occupancy(const cudaDeviceProp &device_prop,
                                     double *occupancy_pct,
                                     int *active_blocks_per_sm) {
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, update_from_laplacian, kUpdateBlockX * kUpdateBlockY, 0));
    const int max_warps_per_sm = device_prop.maxThreadsPerMultiProcessor / device_prop.warpSize;
    const int active_warps = active_blocks * ((kUpdateBlockX * kUpdateBlockY + device_prop.warpSize - 1) / device_prop.warpSize);
    *active_blocks_per_sm = active_blocks;
    *occupancy_pct = std::min(100.0, 100.0 * static_cast<double>(active_warps) / static_cast<double>(max_warps_per_sm));
}

void maybe_export_initial_snapshot(bool export_snapshots,
                                   const std::string &snapshot_dir,
                                   const std::string &backend_label,
                                   double domain_length,
                                   int nx,
                                   int ny,
                                   const std::vector<double> &initial_curr) {
    if (!export_snapshots) {
        return;
    }
    write_snapshot_csv(snapshot_dir, backend_label, domain_length, nx, ny, 0, initial_curr);
}

void maybe_export_runtime_snapshot(bool export_snapshots,
                                   const std::string &snapshot_dir,
                                   const std::string &backend_label,
                                   double domain_length,
                                   int nx,
                                   int ny,
                                   int current_step,
                                   const std::vector<int> &selected_steps,
                                   size_t *next_snapshot_index,
                                   const double *d_curr) {
    if (!export_snapshots) {
        return;
    }

    while (*next_snapshot_index < selected_steps.size() &&
           selected_steps[*next_snapshot_index] == current_step) {
        std::vector<double> snapshot(static_cast<size_t>(nx) * static_cast<size_t>(ny), 0.0);
        CUDA_CHECK(cudaMemcpy(snapshot.data(),
                              d_curr,
                              snapshot.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        write_snapshot_csv(snapshot_dir, backend_label, domain_length, nx, ny, current_step, snapshot);
        ++(*next_snapshot_index);
    }
}

RunResult run_cusparse(const Options &options,
                       int nx,
                       int ny,
                       double lambda2,
                       const std::vector<double> &initial_prev,
                       const std::vector<double> &initial_curr,
                       const cudaDeviceProp &device_prop) {
    RunResult result;
    result.backend_label = "B1_cusparse";
    result.nx = nx;
    result.ny = ny;
    result.steps = options.steps;
    compute_update_kernel_occupancy(device_prop, &result.update_occupancy_pct, &result.active_blocks_per_sm);

    const size_t total_points = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t field_bytes = total_points * sizeof(double);
    const CsrMatrixHost csr = build_laplacian_csr(nx, ny);
    const int nnz = static_cast<int>(csr.values.size());

    double *d_prev = nullptr;
    double *d_curr = nullptr;
    double *d_next = nullptr;
    double *d_laplacian = nullptr;
    int *d_row_offsets = nullptr;
    int *d_col_indices = nullptr;
    double *d_values = nullptr;
    void *d_buffer = nullptr;

    auto sim_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_prev), field_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_curr), field_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_next), field_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_laplacian), field_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_row_offsets), csr.row_offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_col_indices), csr.col_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_values), csr.values.size() * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_prev, initial_prev.data(), field_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_curr, initial_curr.data(), field_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_next, 0, field_bytes));
    CUDA_CHECK(cudaMemcpy(d_row_offsets, csr.row_offsets.data(), csr.row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices, csr.col_indices.data(), csr.col_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, csr.values.data(), csr.values.size() * sizeof(double), cudaMemcpyHostToDevice));

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t mat_a = nullptr;
    cusparseDnVecDescr_t vec_x = nullptr;
    cusparseDnVecDescr_t vec_y = nullptr;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    CUSPARSE_CHECK(cusparseCreateCsr(&mat_a,
                                     static_cast<int64_t>(total_points),
                                     static_cast<int64_t>(total_points),
                                     static_cast<int64_t>(nnz),
                                     d_row_offsets,
                                     d_col_indices,
                                     d_values,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_x, static_cast<int64_t>(total_points), d_curr, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y, static_cast<int64_t>(total_points), d_laplacian, CUDA_R_64F));

    const double alpha = 1.0;
    const double beta = 0.0;
    size_t buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           mat_a,
                                           vec_x,
                                           &beta,
                                           vec_y,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_ALG_DEFAULT,
                                           &buffer_size));
    CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));

    const dim3 block(kUpdateBlockX, kUpdateBlockY);
    const dim3 grid((static_cast<unsigned int>(nx) + block.x - 1) / block.x,
                    (static_cast<unsigned int>(ny) + block.y - 1) / block.y);

    const std::vector<int> selected_steps = build_snapshot_steps(options.steps);
    size_t next_snapshot_index = 0;
    if (!selected_steps.empty() && selected_steps[0] == 0) {
        maybe_export_initial_snapshot(options.export_snapshots,
                                      options.snapshot_dir,
                                      result.backend_label,
                                      options.domain_length,
                                      nx,
                                      ny,
                                      initial_curr);
        next_snapshot_index = 1;
    }

    cudaEvent_t gpu_start;
    cudaEvent_t gpu_stop;
    cudaEvent_t lib_start;
    cudaEvent_t lib_stop;
    CUDA_CHECK(cudaEventCreate(&gpu_start));
    CUDA_CHECK(cudaEventCreate(&gpu_stop));
    CUDA_CHECK(cudaEventCreate(&lib_start));
    CUDA_CHECK(cudaEventCreate(&lib_stop));

    CUDA_CHECK(cudaEventRecord(gpu_start));
    for (int step = 0; step < options.steps; ++step) {
        CUSPARSE_CHECK(cusparseDnVecSetValues(vec_x, d_curr));
        CUSPARSE_CHECK(cusparseDnVecSetValues(vec_y, d_laplacian));

        CUDA_CHECK(cudaEventRecord(lib_start));
        CUSPARSE_CHECK(cusparseSpMV(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    mat_a,
                                    vec_x,
                                    &beta,
                                    vec_y,
                                    CUDA_R_64F,
                                    CUSPARSE_SPMV_ALG_DEFAULT,
                                    d_buffer));
        CUDA_CHECK(cudaEventRecord(lib_stop));
        CUDA_CHECK(cudaEventSynchronize(lib_stop));
        float lib_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&lib_ms, lib_start, lib_stop));
        result.library_ms_total += lib_ms;

        update_from_laplacian<<<grid, block>>>(d_prev, d_curr, d_laplacian, d_next, nx, ny, lambda2);
        double *tmp = d_prev;
        d_prev = d_curr;
        d_curr = d_next;
        d_next = tmp;

        maybe_export_runtime_snapshot(options.export_snapshots,
                                      options.snapshot_dir,
                                      result.backend_label,
                                      options.domain_length,
                                      nx,
                                      ny,
                                      step + 1,
                                      selected_steps,
                                      &next_snapshot_index,
                                      d_curr);
    }
    CUDA_CHECK(cudaEventRecord(gpu_stop));
    CUDA_CHECK(cudaEventSynchronize(gpu_stop));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventElapsedTime(&result.gpu_ms_total, gpu_start, gpu_stop));
    result.gpu_ms_per_step = result.gpu_ms_total / static_cast<float>(options.steps);
    result.library_ms_per_step = result.library_ms_total / static_cast<float>(options.steps);

    result.final_field.assign(total_points, 0.0);
    CUDA_CHECK(cudaMemcpy(result.final_field.data(), d_curr, field_bytes, cudaMemcpyDeviceToHost));
    auto sim_stop = std::chrono::steady_clock::now();
    result.sim_ms_total = std::chrono::duration<double, std::milli>(sim_stop - sim_start).count();
    result.checksum = compute_checksum(result.final_field);
    result.completed = true;

    const double interior_updates = static_cast<double>(std::max(nx - 2, 0)) * static_cast<double>(std::max(ny - 2, 0));
    const double total_bytes = kBytesPerUpdate * interior_updates * static_cast<double>(options.steps);
    result.bandwidth_gbps = total_bytes / (static_cast<double>(result.gpu_ms_total) * 1.0e6);

    CUDA_CHECK(cudaEventDestroy(gpu_start));
    CUDA_CHECK(cudaEventDestroy(gpu_stop));
    CUDA_CHECK(cudaEventDestroy(lib_start));
    CUDA_CHECK(cudaEventDestroy(lib_stop));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_x));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_y));
    CUSPARSE_CHECK(cusparseDestroySpMat(mat_a));
    CUSPARSE_CHECK(cusparseDestroy(handle));
    CUDA_CHECK(cudaFree(d_buffer));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_row_offsets));
    CUDA_CHECK(cudaFree(d_laplacian));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_curr));
    CUDA_CHECK(cudaFree(d_prev));

    return result;
}

RunResult run_cublas(const Options &options,
                     int nx,
                     int ny,
                     double lambda2,
                     const std::vector<double> &initial_prev,
                     const std::vector<double> &initial_curr,
                     const cudaDeviceProp &device_prop) {
    RunResult result;
    result.backend_label = "B2_cublas";
    result.nx = nx;
    result.ny = ny;
    result.steps = options.steps;
    compute_update_kernel_occupancy(device_prop, &result.update_occupancy_pct, &result.active_blocks_per_sm);

    const size_t total_points = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t field_bytes = total_points * sizeof(double);
    size_t dense_bytes = 0;
    if (!estimate_dense_matrix_bytes(total_points, &dense_bytes)) {
        result.note = "dense_matrix_size_overflow";
        return result;
    }

    size_t free_mem_bytes = 0;
    size_t total_mem_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem_bytes, &total_mem_bytes));
    if (!dense_matrix_fits_budget(dense_bytes, options.dense_max_gb, free_mem_bytes)) {
        result.note = "dense_matrix_exceeds_budget";
        return result;
    }

    DenseMatrixHost dense;
    try {
        dense = build_laplacian_dense_column_major(nx, ny);
    } catch (const std::bad_alloc &) {
        result.note = "host_dense_allocation_failed";
        return result;
    }

    double *d_prev = nullptr;
    double *d_curr = nullptr;
    double *d_next = nullptr;
    double *d_laplacian = nullptr;
    double *d_dense = nullptr;

    auto sim_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_prev), field_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_curr), field_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_next), field_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_laplacian), field_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_dense), dense.bytes));

    CUDA_CHECK(cudaMemcpy(d_prev, initial_prev.data(), field_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_curr, initial_curr.data(), field_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_next, 0, field_bytes));
    CUDA_CHECK(cudaMemcpy(d_dense, dense.values.data(), dense.bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));

    const dim3 block(kUpdateBlockX, kUpdateBlockY);
    const dim3 grid((static_cast<unsigned int>(nx) + block.x - 1) / block.x,
                    (static_cast<unsigned int>(ny) + block.y - 1) / block.y);

    const std::vector<int> selected_steps = build_snapshot_steps(options.steps);
    size_t next_snapshot_index = 0;
    if (!selected_steps.empty() && selected_steps[0] == 0) {
        maybe_export_initial_snapshot(options.export_snapshots,
                                      options.snapshot_dir,
                                      result.backend_label,
                                      options.domain_length,
                                      nx,
                                      ny,
                                      initial_curr);
        next_snapshot_index = 1;
    }

    const double alpha = 1.0;
    const double beta = 0.0;

    cudaEvent_t gpu_start;
    cudaEvent_t gpu_stop;
    cudaEvent_t lib_start;
    cudaEvent_t lib_stop;
    CUDA_CHECK(cudaEventCreate(&gpu_start));
    CUDA_CHECK(cudaEventCreate(&gpu_stop));
    CUDA_CHECK(cudaEventCreate(&lib_start));
    CUDA_CHECK(cudaEventCreate(&lib_stop));

    CUDA_CHECK(cudaEventRecord(gpu_start));
    for (int step = 0; step < options.steps; ++step) {
        CUDA_CHECK(cudaEventRecord(lib_start));
        CUBLAS_CHECK(cublasDgemv(handle,
                                 CUBLAS_OP_N,
                                 static_cast<int>(total_points),
                                 static_cast<int>(total_points),
                                 &alpha,
                                 d_dense,
                                 static_cast<int>(total_points),
                                 d_curr,
                                 1,
                                 &beta,
                                 d_laplacian,
                                 1));
        CUDA_CHECK(cudaEventRecord(lib_stop));
        CUDA_CHECK(cudaEventSynchronize(lib_stop));
        float lib_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&lib_ms, lib_start, lib_stop));
        result.library_ms_total += lib_ms;

        update_from_laplacian<<<grid, block>>>(d_prev, d_curr, d_laplacian, d_next, nx, ny, lambda2);
        double *tmp = d_prev;
        d_prev = d_curr;
        d_curr = d_next;
        d_next = tmp;

        maybe_export_runtime_snapshot(options.export_snapshots,
                                      options.snapshot_dir,
                                      result.backend_label,
                                      options.domain_length,
                                      nx,
                                      ny,
                                      step + 1,
                                      selected_steps,
                                      &next_snapshot_index,
                                      d_curr);
    }
    CUDA_CHECK(cudaEventRecord(gpu_stop));
    CUDA_CHECK(cudaEventSynchronize(gpu_stop));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventElapsedTime(&result.gpu_ms_total, gpu_start, gpu_stop));
    result.gpu_ms_per_step = result.gpu_ms_total / static_cast<float>(options.steps);
    result.library_ms_per_step = result.library_ms_total / static_cast<float>(options.steps);

    result.final_field.assign(total_points, 0.0);
    CUDA_CHECK(cudaMemcpy(result.final_field.data(), d_curr, field_bytes, cudaMemcpyDeviceToHost));
    auto sim_stop = std::chrono::steady_clock::now();
    result.sim_ms_total = std::chrono::duration<double, std::milli>(sim_stop - sim_start).count();
    result.checksum = compute_checksum(result.final_field);
    result.completed = true;

    const double interior_updates = static_cast<double>(std::max(nx - 2, 0)) * static_cast<double>(std::max(ny - 2, 0));
    const double total_bytes = kBytesPerUpdate * interior_updates * static_cast<double>(options.steps);
    result.bandwidth_gbps = total_bytes / (static_cast<double>(result.gpu_ms_total) * 1.0e6);

    CUDA_CHECK(cudaEventDestroy(gpu_start));
    CUDA_CHECK(cudaEventDestroy(gpu_stop));
    CUDA_CHECK(cudaEventDestroy(lib_start));
    CUDA_CHECK(cudaEventDestroy(lib_stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_dense));
    CUDA_CHECK(cudaFree(d_laplacian));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_curr));
    CUDA_CHECK(cudaFree(d_prev));

    return result;
}

void print_result_line(const RunResult &result, double domain_length) {
    printf(
        "RESULT backend=%s length=%.2f nx=%d ny=%d steps=%d "
        "gpu_ms_total=%.6f gpu_ms_step=%.6f "
        "library_ms_total=%.6f library_ms_step=%.6f "
        "sim_ms_total=%.6f bandwidth_GBps=%.6f "
        "update_occupancy_pct=%.2f active_blocks_per_sm=%d "
        "checksum=%.12e max_error=%.12e note=%s\n",
        result.backend_label.c_str(),
        domain_length,
        result.nx,
        result.ny,
        result.steps,
        result.gpu_ms_total,
        result.gpu_ms_per_step,
        result.library_ms_total,
        result.library_ms_per_step,
        result.sim_ms_total,
        result.bandwidth_gbps,
        result.update_occupancy_pct,
        result.active_blocks_per_sm,
        result.checksum,
        result.max_error,
        result.note.empty() ? "none" : result.note.c_str());
}

void print_skip_line(const RunResult &result, double domain_length) {
    printf("SKIP backend=%s length=%.2f nx=%d ny=%d steps=%d note=%s\n",
           result.backend_label.c_str(),
           domain_length,
           result.nx,
           result.ny,
           result.steps,
           result.note.empty() ? "unspecified" : result.note.c_str());
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

    printf("DEVICE name=\"%s\" sm_count=%d warp_size=%d max_threads_per_sm=%d shared_mem_per_sm=%zu total_global_mem=%zu\n",
           device_prop.name,
           device_prop.multiProcessorCount,
           device_prop.warpSize,
           device_prop.maxThreadsPerMultiProcessor,
           device_prop.sharedMemPerMultiprocessor,
           device_prop.totalGlobalMem);
    printf("CONFIG steps=%d run_cusparse=%d run_cublas=%d export_snapshots=%d dense_max_gb=%.2f\n",
           options.steps,
           kRunCuSparse ? 1 : 0,
           kRunCuBlas ? 1 : 0,
           options.export_snapshots ? 1 : 0,
           options.dense_max_gb);
    printf("MEASURE bytes_per_update=%.0f update_block=%dx%d\n",
           kBytesPerUpdate,
           kUpdateBlockX,
           kUpdateBlockY);

    if (kRunCuSparse) {
        for (double domain_length : kCuSparseLengths) {
            const int nx = grid_points_from_length(domain_length, kDx);
            const int ny = grid_points_from_length(domain_length, kDy);
            std::vector<double> initial_prev;
            std::vector<double> initial_curr;
            initialize_fields(&initial_prev, &initial_curr, nx, ny, kDx, kDy, lambda2);
            Options run_options = options;
            run_options.domain_length = domain_length;

            printf("SETUP backend=B1_cusparse length=%.2f dx=%.5f dy=%.5f dt=%.5f c=%.2f lambda=%.6f lambda2=%.6f nx=%d ny=%d steps=%d\n",
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

            RunResult cusparse_result = run_cusparse(
                run_options, nx, ny, lambda2, initial_prev, initial_curr, device_prop);
            print_result_line(cusparse_result, domain_length);
        }
    }

    if (kRunCuBlas) {
        for (double domain_length : kCuBlasLengths) {
            const int nx = grid_points_from_length(domain_length, kDx);
            const int ny = grid_points_from_length(domain_length, kDy);
            std::vector<double> initial_prev;
            std::vector<double> initial_curr;
            initialize_fields(&initial_prev, &initial_curr, nx, ny, kDx, kDy, lambda2);
            Options run_options = options;
            run_options.domain_length = domain_length;

            printf("SETUP backend=B2_cublas length=%.2f dx=%.5f dy=%.5f dt=%.5f c=%.2f lambda=%.6f lambda2=%.6f nx=%d ny=%d steps=%d\n",
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

            RunResult cublas_result = run_cublas(
                run_options, nx, ny, lambda2, initial_prev, initial_curr, device_prop);
            if (!cublas_result.completed) {
                print_skip_line(cublas_result, domain_length);
            } else {
                print_result_line(cublas_result, domain_length);
            }
        }
    }

    return 0;
}
