#pragma once

#include <vector>

#include "fires.hpp"
#include "landscape.hpp"
#include "spread_functions.hpp"

/* Make `n_replicates` simulation and return a matrix with the number of simulations each cell
 * was burned.
 */
Matrix<size_t> burned_amounts_per_cell(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, float distance, float elevation_mean, float elevation_sd,
    float upper_limit, size_t n_replicates
);
