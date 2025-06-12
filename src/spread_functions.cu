#include "spread_functions.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <iostream>

#include "fires.hpp"
#include "landscape.hpp"
#include <iostream>

// CUDA specific includes
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <omp.h>


// Helper for CUDA error checking
#define CUDA_CHECK(err) { \
    cudaError_t e = err; \
    if (e != cudaSuccess) { \
        printf("Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}


__global__ void setup_rand_states(curandState* states, unsigned long seed, size_t total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Keep the host-side function for CPU execution if needed
float spread_probability_host(
    const Cell& burning, const Cell& neighbour, SimulationParams params, float angle,
    float distance, float elevation_mean, float elevation_sd, float upper_limit = 1.0
);

// Device-side version of the spread probability function
__device__ float spread_probability_device(
    const Cell& burning, const Cell& neighbour, const SimulationParams& params, float angle,
    float distance, float elevation_mean, float elevation_sd, float upper_limit = 1.0
) {
    // Note: Using device-side math functions (e.g., sinf, cosf, expf)
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
    const Cell* d_cells,         // All landscape cells (device)
    unsigned int* d_burned_bin,  // CORRECTED: Changed from bool* to unsigned int*
    int2* d_burned_ids,          // Array of currently burning/burned cell coords (device)
    curandState* d_rand_states,  // cuRAND states for each thread (device)
    unsigned int* d_newly_burned_count, // Atomic counter for new fires (device)
    unsigned int* d_contador, // Added: Atomic counter for total neighbor evaluations
    const SimulationParams params, // Simulation parameters
    size_t width, size_t height,
    size_t start, size_t end,
    float distance, float elevation_mean, float elevation_sd, float upper_limit
) {
    // Calculate the global thread ID for the current work window [start, end)
    unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int work_size = end - start;

    if (thread_idx >= work_size) {
        return;
    }
    
    // The actual index in the d_burned_ids array
    unsigned int burn_id_idx = start + thread_idx;

    // Get the coordinates of the cell this thread is processing
    int2 burning_cell_coords = d_burned_ids[burn_id_idx];
    const Cell& burning_cell = d_cells[burning_cell_coords.y * width + burning_cell_coords.x];

    // Get this thread's random state
    curandState* rand_state = &d_rand_states[thread_idx];

    // Define neighbor moves and corresponding angles
    constexpr int moves[8][2] = { { -1, -1 }, { -1,  0 }, { -1, 1 }, { 0, -1 },
                                  {  0,  1 }, {  1, -1 }, {  1, 0 }, { 1,  1 } };
    constexpr float angles[8] = { 3.14159265f * 3 / 4, 3.14159265f,     3.14159265f * 5 / 4, 3.14159265f / 2,
                                  3.14159265f * 3 / 2, 3.14159265f / 4, 0,                 3.14159265f * 7 / 4 };

    // Loop over the 8 neighbors
    for (int n = 0; n < 8; ++n) {
        // Increment the global counter for each neighbor evaluated
        atomicAdd(d_contador, 1); 

        int2 neighbour_coords = { burning_cell_coords.x + moves[n][0], burning_cell_coords.y + moves[n][1] };

        // Check if the neighbor is within the landscape boundaries
        if (neighbour_coords.x < 0 || neighbour_coords.x >= (int)width || neighbour_coords.y < 0 || neighbour_coords.y >= (int)height) {
            continue;
        }

        size_t neighbour_flat_idx = neighbour_coords.y * width + neighbour_coords.x;
        const Cell& neighbour_cell = d_cells[neighbour_flat_idx];

        // Atomically check and set the burned status.
        // We now directly use d_burned_bin as it's already the correct type.
        if (neighbour_cell.burnable && atomicCAS(&d_burned_bin[neighbour_flat_idx], 0, 1) == 0) {
            
            float prob = spread_probability_device(
                burning_cell, neighbour_cell, params, angles[n], distance,
                elevation_mean, elevation_sd, upper_limit
            );

            // Burn with probability (Bernoulli trial)
            if (curand_uniform(rand_state) < prob) {
                // Get a unique index to write the new burnable cell to.
                unsigned int write_idx = atomicAdd(d_newly_burned_count, 1);
                
                // Write the coordinates of the newly burned cell at the end of the list.
                d_burned_ids[end + write_idx] = neighbour_coords;
            } else {
                // If it didn't burn, revert the burned_bin status.
                atomicExch(&d_burned_bin[neighbour_flat_idx], 0);
            }
        }
    }
}

// New host function to run the simulation on the GPU
Fire simulate_fire(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, float distance, float elevation_mean, float elevation_sd,
    int &h_contador, float upper_limit = 1.0
) {
    size_t n_row = landscape.height;
    size_t n_col = landscape.width;
    size_t total_cells = n_row * n_col;

    // --- 1. Allocate Host Memory ---
    // This host vector is used for transferring burn data to/from the GPU.
    // We use 'unsigned int' to match the size for atomic operations on the GPU.
    std::vector<unsigned int> h_burned_bin_transfer(total_cells, 0);
    
    std::vector<int2> h_burned_ids(total_cells); // Max possible size
    std::vector<size_t> burned_ids_steps;

    // Copy initial ignition points
    for (size_t i = 0; i < ignition_cells.size(); ++i) {
        const auto& coords = ignition_cells[i];
        h_burned_ids[i] = { (int)coords.first, (int)coords.second };
        h_burned_bin_transfer[coords.second * n_col + coords.first] = 1;
    }

    // --- 2. Allocate Device Memory ---
    Cell* d_cells;
    unsigned int* d_burned_bin; // Use unsigned int to match the host transfer vector
    int2* d_burned_ids;
    curandState* d_rand_states;
    unsigned int* d_newly_burned_count;
    unsigned int* d_contador; // Added: Device pointer for contador

    CUDA_CHECK(cudaMalloc(&d_cells, total_cells * sizeof(Cell)));
    CUDA_CHECK(cudaMalloc(&d_burned_bin, total_cells * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_burned_ids, total_cells * sizeof(int2)));
    CUDA_CHECK(cudaMalloc(&d_newly_burned_count, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_rand_states, total_cells * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_contador, sizeof(unsigned int))); // Allocate device memory for contador
    

    // --- 3. Copy Data from Host to Device ---
    // Access the underlying vector's contiguous data for the Cell matrix
    CUDA_CHECK(cudaMemcpy(d_cells, landscape.cells.elems.data(), total_cells * sizeof(Cell), cudaMemcpyHostToDevice));
    
    // Copy from our contiguous transfer vector
    CUDA_CHECK(cudaMemcpy(d_burned_bin, h_burned_bin_transfer.data(), total_cells * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_burned_ids, h_burned_ids.data(), ignition_cells.size() * sizeof(int2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_contador, 0, sizeof(unsigned int))); // Initialize device contador to 0

    // --- 4. Simulation Loop ---
    size_t start = 0;
    size_t end = ignition_cells.size();
    burned_ids_steps.push_back(end);
    
    int threads_per_block = 256;

    // Setup random states
    int total_threads = total_cells; // Or the max number of threads you will use
    int setup_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    setup_rand_states<<<setup_blocks, threads_per_block>>>(d_rand_states, 1234, total_threads);
    CUDA_CHECK(cudaDeviceSynchronize());

    double t = omp_get_wtime();

    while (end > start) {
        size_t work_size = end - start;
        int blocks = (work_size + threads_per_block - 1) / threads_per_block;

        CUDA_CHECK(cudaMemset(d_newly_burned_count, 0, sizeof(unsigned int)));
        
        evaluate_spread_kernel<<<blocks, threads_per_block>>>(
            d_cells, d_burned_bin, d_burned_ids, d_rand_states,
            d_newly_burned_count, d_contador, // Passed d_contador to the kernel
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
    
    // --- 5. Copy Results Back to Host ---
    size_t final_burned_count = end;
    h_burned_ids.resize(final_burned_count);
    
    // Copy results back to the contiguous transfer vector
    CUDA_CHECK(cudaMemcpy(h_burned_bin_transfer.data(), d_burned_bin, total_cells * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_burned_ids.data(), d_burned_ids, final_burned_count * sizeof(int2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_contador, d_contador, sizeof(unsigned int), cudaMemcpyDeviceToHost)); // Copy contador back to host

    // --- 6. Free Device Memory ---
    CUDA_CHECK(cudaFree(d_cells));
    CUDA_CHECK(cudaFree(d_burned_bin));
    CUDA_CHECK(cudaFree(d_burned_ids));
    CUDA_CHECK(cudaFree(d_rand_states));
    CUDA_CHECK(cudaFree(d_newly_burned_count));
    CUDA_CHECK(cudaFree(d_contador)); // Free device memory for contador

    // --- 7. Convert results to the final 'Fire' struct format ---
    Matrix<bool> final_burned_bin(n_col, n_row);
    // Manually populate the Matrix<bool> from the transfer vector
    for(size_t r = 0; r < n_row; ++r) {
        for(size_t c = 0; c < n_col; ++c) {
            final_burned_bin[{c, r}] = (h_burned_bin_transfer[r * n_col + c] == 1);
        }
    }
    
    std::vector<std::pair<size_t, size_t>> final_burned_ids(final_burned_count);
    for(size_t i = 0; i < final_burned_count; ++i) {
        final_burned_ids[i] = { (size_t)h_burned_ids[i].x, (size_t)h_burned_ids[i].y };
    }

    double t2 = omp_get_wtime() - t;
    std::cerr << "Celdas/ms: " << h_contador*1000/(t2) << std::endl;


    return { n_col, n_row, final_burned_bin, final_burned_ids, burned_ids_steps };
}
