#include "many_simulations.hpp"

#include <cmath>

Matrix<size_t> burned_amounts_per_cell(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, float distance, float elevation_mean, float elevation_sd,
    float upper_limit, size_t n_replicates
) {

  Matrix<size_t> burned_amounts(landscape.width, landscape.height);

  for (size_t i = 0; i < n_replicates; i++) {
    int contador = 0;
    Fire fire = simulate_fire(
        landscape, ignition_cells, params, distance, elevation_mean, elevation_sd, contador, upper_limit
    );

    for (size_t col = 0; col < landscape.width; col++) {
      for (size_t row = 0; row < landscape.height; row++) {
        burned_amounts[{col, row}] += fire.burned_layer[{col, row}];
      }
    }
  }

  return burned_amounts;
}
