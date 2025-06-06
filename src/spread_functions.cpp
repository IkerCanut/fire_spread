#include "spread_functions.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <random>
#include <vector>
#include <utility>

#include "fires.hpp"
#include "landscape.hpp"
#include <iostream>
#include <omp.h>
#include <immintrin.h>

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

Fire simulate_fire(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, float distance, float elevation_mean, float elevation_sd, int &contador,
    float upper_limit = 1.0
) {

  std::vector<std::mt19937> rng_per_thread;
  std::vector<std::uniform_real_distribution<float>> dist_per_thread;
  unsigned int initial_seed = std::random_device{}();
  int max_threads = omp_get_max_threads();
  for (int i = 0; i < max_threads; ++i) {
    rng_per_thread.emplace_back(initial_seed + i);
    dist_per_thread.emplace_back(0.0f, 1.0f);
  }


  size_t n_row = landscape.height;
  size_t n_col = landscape.width;

  std::vector<std::pair<size_t, size_t>> burned_ids;

  size_t start = 0;
  size_t end = ignition_cells.size();

  for (size_t i = 0; i < end; i++) {
    burned_ids.push_back(ignition_cells[i]);
  }

  std::vector<size_t> burned_ids_steps;
  burned_ids_steps.push_back(end);

  size_t burning_size = end + 1;

  Matrix<bool> burned_bin = Matrix<bool>(n_col, n_row);

  for (size_t i = 0; i < end; i++) {
    burned_bin[ignition_cells[i]] = 1;
  }

  double t = omp_get_wtime();
  while (burning_size > 0) {
    size_t end_forward = end;

    // Loop over burning cells in the cycle

    // b is going to keep the position in burned_ids that have to be evaluated
    // in this burn cycle
    for (size_t b = start; b < end; b++) {
      size_t burning_cell_0 = burned_ids[b].first;
      size_t burning_cell_1 = burned_ids[b].second;

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

      #pragma omp parallel for reduction(+:contador) \
          shared(landscape, burning_cell, params, angles, distance, elevation_mean, elevation_sd, upper_limit, \
                 n_col, n_row, neighbors_coords, burned_ids, burned_bin, end_forward, \
                 rng_per_thread, dist_per_thread) \
          default(none)
      for (size_t n = 0; n < 8; n++) {
        contador++;

        int neighbour_cell_0 = neighbors_coords[0][n];
        int neighbour_cell_1 = neighbors_coords[1][n];

        // Is the cell in range?
        bool out_of_range = 0 > neighbour_cell_0 || neighbour_cell_0 >= int(n_col) ||
                            0 > neighbour_cell_1 || neighbour_cell_1 >= int(n_row);

        if (out_of_range)
          continue;

        const Cell& neighbour_cell = landscape[{ neighbour_cell_0, neighbour_cell_1 }];

        // Is the cell burnable?
        bool burnable_cell =
            !burned_bin[{ neighbour_cell_0, neighbour_cell_1 }] && neighbour_cell.burnable;

        if (!burnable_cell)
          continue;

        // simulate fire
        float prob = spread_probability(
            burning_cell, neighbour_cell, params, angles[n], distance, elevation_mean,
            elevation_sd, upper_limit
        );

        // Burn with probability prob (Bernoulli)
        int thread_num = omp_get_thread_num();
        bool burn = dist_per_thread[thread_num](rng_per_thread[thread_num]) < prob;

        if (burn == 0)
          continue;

        #pragma omp critical
        {
          end_forward += 1;
          burned_ids.push_back({ neighbour_cell_0, neighbour_cell_1 });
        }
        burned_bin[{ neighbour_cell_0, neighbour_cell_1 }] = true;
      }
    }

    // update start and end
    start = end;
    end = end_forward;
    burning_size = end - start;

    burned_ids_steps.push_back(end);
  }
  double t2 = omp_get_wtime() - t;
  std::cerr << "Celdas/ms: " << contador*1000/(t2) << std::endl;

  return { n_col, n_row, burned_bin, burned_ids, burned_ids_steps };
}
