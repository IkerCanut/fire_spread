#include "spread_functions.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <iostream>

#include "fires.hpp"
#include "landscape.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <omp.h>
#include <immintrin.h>
#include <random>


#define THRESHOLD 32

#define CUDA_CHECK(err) { \
    cudaError_t e = err; \
    if (e != cudaSuccess) { \
        printf("Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

alignas(32) constexpr int moves_x[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
alignas(32) constexpr int moves_y[8] = { -1,  0,  1, -1, 1, -1, 0, 1 };

constexpr float angles[8] = { M_PI * 3 / 4, M_PI,     M_PI * 5 / 4, M_PI / 2,
  M_PI * 3 / 2, M_PI / 4, 0,            M_PI * 7 / 4 };

float spread_probability(
    const Cell& burning, const Cell& neighbour, SimulationParams params, float angle,
    float distance, float elevation_mean, float elevation_sd, float upper_limit = 1.0
) {

  float slope_term = sin(atan((neighbour.elevation - burning.elevation) / distance));
  float wind_term = cos(angle - burning.wind_direction);
  float elev_term = (neighbour.elevation - elevation_mean) / elevation_sd;

  float linpred = params.independent_pred;

  if (neighbour.vegetation_type == SUBALPINE) {
    linpred += params.subalpine_pred;
  } else if (neighbour.vegetation_type == WET) {
    linpred += params.wet_pred;
  } else if (neighbour.vegetation_type == DRY) {
    linpred += params.dry_pred;
  }

  linpred += params.fwi_pred * neighbour.fwi;
  linpred += params.aspect_pred * neighbour.aspect;

  linpred += wind_term * params.wind_pred + elev_term * params.elevation_pred +
             slope_term * params.slope_pred;

  float prob = upper_limit / (1 + exp(-linpred));

  return prob;
}

__global__ void setup_rand_states(curandState* states, unsigned long seed, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__device__ float spread_probability_device(
    const Cell& burning, const Cell& neighbour, const SimulationParams& params, float angle,
    float distance, float elevation_mean, float elevation_sd, float upper_limit = 1.0
) {
    float slope_term = sinf(atanf((neighbour.elevation - burning.elevation) / distance));
    float wind_term = cosf(angle - burning.wind_direction);
    float elev_term = (neighbour.elevation - elevation_mean) / elevation_sd;

    float linpred = params.independent_pred;

    if (neighbour.vegetation_type == SUBALPINE) {
        linpred += params.subalpine_pred;
    } else if (neighbour.vegetation_type == WET) {
        linpred += params.wet_pred;
    } else if (neighbour.vegetation_type == DRY) {
        linpred += params.dry_pred;
    }

    linpred += params.fwi_pred * neighbour.fwi;
    linpred += params.aspect_pred * neighbour.aspect;

    linpred += wind_term * params.wind_pred + elev_term * params.elevation_pred +
               slope_term * params.slope_pred;

    float prob = upper_limit / (1.0f + expf(-linpred));

    return prob;
}


__global__ void evaluate_spread_kernel(
    const Cell* d_cells,
    unsigned int* d_burned_bin,
    int2* d_burned_ids,
    curandState* d_rand_states,
    unsigned int* d_newly_burned_count,
    unsigned int* d_contador,
    const SimulationParams params,
    size_t width, size_t height,
    size_t start, size_t end,
    float distance, float elevation_mean, float elevation_sd, float upper_limit
) {
    unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int work_size = end - start;

    if (thread_idx >= work_size) {
        return;
    }
    
    unsigned int burn_id_idx = start + thread_idx;

    int2 burning_cell_coords = d_burned_ids[burn_id_idx];
    const Cell& burning_cell = d_cells[burning_cell_coords.y * width + burning_cell_coords.x];

    curandState* rand_state = &d_rand_states[thread_idx];

    constexpr int moves[8][2] = { { -1, -1 }, { -1,  0 }, { -1, 1 }, { 0, -1 },
                                  {  0,  1 }, {  1, -1 }, {  1, 0 }, { 1,  1 } };
    constexpr float angles[8] = { M_PI * 3 / 4, M_PI    , M_PI * 5 / 4, M_PI / 2,
                                  M_PI * 3 / 2, M_PI / 4, 0,            M_PI * 7 / 4 };

    size_t tid  = threadIdx.x;      // thread id, dentro del bloque
    __shared__ int suma_par;

    if (tid==0)
        suma_par = 0;
    __syncthreads();

    for (int n = 0; n < 8; ++n) {
        atomicAdd(&suma_par, 1); 

        int2 neighbour_coords = { burning_cell_coords.x + moves[n][0], burning_cell_coords.y + moves[n][1] };

        if (neighbour_coords.x < 0 || neighbour_coords.x >= (int)width ||
            neighbour_coords.y < 0 || neighbour_coords.y >= (int)height) {
            continue;
        }

        size_t neighbour_flat_idx = neighbour_coords.y * width + neighbour_coords.x;
        const Cell& neighbour_cell = d_cells[neighbour_flat_idx];

        if (neighbour_cell.burnable && atomicCAS(&d_burned_bin[neighbour_flat_idx], 0, 1) == 0) {
            float prob = spread_probability_device(
                burning_cell, neighbour_cell, params, angles[n], distance,
                elevation_mean, elevation_sd, upper_limit
            );
            if (curand_uniform(rand_state) < prob) {
                unsigned int write_idx = atomicAdd(d_newly_burned_count, 1);
                d_burned_ids[end + write_idx] = neighbour_coords;
            } else {
                atomicExch(&d_burned_bin[neighbour_flat_idx], 0);
            }
        }
    }

    __syncthreads();
    if(tid == 0)
        atomicAdd(d_contador, suma_par); 
}

Fire simulate_fire(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, float distance, float elevation_mean, float elevation_sd,
    int &h_contador, float upper_limit = 1.0
) {
    size_t n_row = landscape.height;
    size_t n_col = landscape.width;
    size_t total_cells = n_row * n_col;

    std::vector<unsigned int> h_burned_bin_transfer(total_cells, 0);
    
    std::vector<int2> h_burned_ids(total_cells);
    std::vector<size_t> burned_ids_steps;

    for (size_t i = 0; i < ignition_cells.size(); ++i) {
        const auto& coords = ignition_cells[i];
        h_burned_ids[i] = { (int)coords.first, (int)coords.second };
        h_burned_bin_transfer[coords.second * n_col + coords.first] = 1;
    }

    Cell* d_cells;
    unsigned int* d_burned_bin;
    int2* d_burned_ids;
    curandState* d_rand_states;
    unsigned int* d_newly_burned_count;
    unsigned int* d_contador;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> uniform_dist(0.0, 1.0);


    CUDA_CHECK(cudaMalloc(&d_cells, total_cells * sizeof(Cell)));
    CUDA_CHECK(cudaMalloc(&d_burned_bin, total_cells * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_burned_ids, total_cells * sizeof(int2)));
    CUDA_CHECK(cudaMalloc(&d_newly_burned_count, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_rand_states, total_cells * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_contador, sizeof(unsigned int)));
    
    CUDA_CHECK(cudaMemcpy(d_cells, landscape.cells.elems.data(), total_cells * sizeof(Cell), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpy(d_burned_bin, h_burned_bin_transfer.data(), total_cells * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_burned_ids, h_burned_ids.data(), ignition_cells.size() * sizeof(int2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_contador, 0, sizeof(unsigned int)));

    size_t start = 0;
    size_t end = ignition_cells.size();
    burned_ids_steps.push_back(end);
    
    int threads_per_block = 512;

    int total_threads = total_cells;
    int setup_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    setup_rand_states<<<setup_blocks, threads_per_block>>>(d_rand_states, 1234, total_threads);
    CUDA_CHECK(cudaDeviceSynchronize());

    double t = omp_get_wtime();

    if (end - start > THRESHOLD) {

        while (start < end) {
            size_t work_size = end - start;
            int blocks = (work_size + threads_per_block - 1) / threads_per_block;
            
            CUDA_CHECK(cudaMemset(d_newly_burned_count, 0, sizeof(unsigned int)));
            
            evaluate_spread_kernel<<<blocks, threads_per_block>>>(
                d_cells, d_burned_bin, d_burned_ids, d_rand_states,
                d_newly_burned_count, d_contador,
                params, n_col, n_row,
                start, end, distance, elevation_mean, elevation_sd, upper_limit
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            
            unsigned int newly_burned_host = 0;
            CUDA_CHECK(cudaMemcpy(&newly_burned_host, d_newly_burned_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            
            start = end;
            end += newly_burned_host;
            
            burned_ids_steps.push_back(end);
        }
    } else {
        size_t end_forward = end;
        for (size_t b = start; b < end; b++) {
            size_t burning_cell_0 = d_burned_ids[b].x;
            size_t burning_cell_1 = d_burned_ids[b].y;

            const Cell& burning_cell = landscape[{ burning_cell_0, burning_cell_1 }];

            int neighbors_coords[2][8];

            __m256i bc0 = _mm256_set1_epi32(int(burning_cell_0));
            __m256i bc1 = _mm256_set1_epi32(int(burning_cell_1));
            __m256i mvx = _mm256_load_si256((__m256i*)moves_x);
            __m256i mvy = _mm256_load_si256((__m256i*)moves_y);

            __m256i nc0 = _mm256_add_epi32(bc0, mvx);
            __m256i nc1 = _mm256_add_epi32(bc1, mvy);

            _mm256_storeu_si256((__m256i*)neighbors_coords[0], nc0);
            _mm256_storeu_si256((__m256i*)neighbors_coords[1], nc1);
            // ---------------------------------------------------

            for (size_t n = 0; n < 8; n++) {
                h_contador++;

                int neighbour_cell_0 = neighbors_coords[0][n];
                int neighbour_cell_1 = neighbors_coords[1][n];

                // Is the cell in range?
                bool out_of_range = 0 > neighbour_cell_0 || neighbour_cell_0 >= int(n_col) ||
                                    0 > neighbour_cell_1 || neighbour_cell_1 >= int(n_row);

                if (out_of_range)
                continue;

                auto burning_cell_coords = d_burned_ids[b];

                int2 neighbour_coords = { burning_cell_coords.x + moves_x[n], burning_cell_coords.y + moves_y[n] };

                if (neighbour_coords.x < 0 || neighbour_coords.x >= (int)n_col ||
                    neighbour_coords.y < 0 || neighbour_coords.y >= (int)n_row) {
                    continue;
                }

                size_t neighbour_flat_idx = neighbour_coords.y * n_col + neighbour_coords.x;

                const Cell& neighbour_cell = landscape[{ neighbour_cell_0, neighbour_cell_1 }];

                // Is the cell burnable?
                bool burnable_cell =
                    !d_burned_bin[neighbour_cell_0 * n_col + neighbour_cell_1] && neighbour_cell.burnable;

                if (!burnable_cell)
                continue;

                // simulate fire
                float prob = spread_probability(
                    burning_cell, neighbour_cell, params, angles[n], distance, elevation_mean,
                    elevation_sd, upper_limit
                );

                // Burn with probability prob (Bernoulli)
                bool burn = uniform_dist(rng) < prob;

                if (burn == 0)
                continue;

                // If burned, store id of recently burned cell and set 1 in burned_bin
                end_forward += 1;
                d_burned_ids[end_forward] = { neighbour_cell_0, neighbour_cell_1 };
                d_burned_bin[{ neighbour_cell_0, neighbour_cell_1 }] = true;
            }
        }
    }
    
    size_t final_burned_count = end;
    h_burned_ids.resize(final_burned_count);
    
    CUDA_CHECK(cudaMemcpy(h_burned_bin_transfer.data(), d_burned_bin, total_cells * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_burned_ids.data(), d_burned_ids, final_burned_count * sizeof(int2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_contador, d_contador, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    double t2 = omp_get_wtime() - t;
    std::cerr << "Celdas/ms: " << h_contador*1000/(t2) << std::endl;
    
    CUDA_CHECK(cudaFree(d_cells));
    CUDA_CHECK(cudaFree(d_burned_bin));
    CUDA_CHECK(cudaFree(d_burned_ids));
    CUDA_CHECK(cudaFree(d_rand_states));
    CUDA_CHECK(cudaFree(d_newly_burned_count));
    CUDA_CHECK(cudaFree(d_contador));

    Matrix<bool> final_burned_bin(n_col, n_row);
    for(size_t r = 0; r < n_row; ++r) {
        for(size_t c = 0; c < n_col; ++c) {
            final_burned_bin[{c, r}] = (h_burned_bin_transfer[r * n_col + c] == 1);
        }
    }
    
    std::vector<std::pair<size_t, size_t>> final_burned_ids(final_burned_count);
    for(size_t i = 0; i < final_burned_count; ++i) {
        final_burned_ids[i] = { (size_t)h_burned_ids[i].x, (size_t)h_burned_ids[i].y };
    }

    return { n_col, n_row, final_burned_bin, final_burned_ids, burned_ids_steps };
}
