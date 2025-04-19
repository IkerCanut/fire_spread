#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

./graphics/burned_probabilities_data ./data/1999_27j_S
