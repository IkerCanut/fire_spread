set -x
make clean
nvcc --use_fast_math -Xcompiler "-Wall -Wextra -Werror -fopenmp" -I./src -c src/csv.cpp -o src/csv.o
nvcc --use_fast_math -Xcompiler "-Wall -Wextra -Werror -fopenmp" -I./src -c src/fires.cpp -o src/fires.o
nvcc --use_fast_math -Xcompiler "-Wall -Wextra -Werror -fopenmp" -I./src -c src/ignition_cells.cpp -o src/ignition_cells.o
nvcc --use_fast_math -Xcompiler "-Wall -Wextra -Werror -fopenmp" -I./src -c src/landscape.cpp -o src/landscape.o
nvcc --use_fast_math -Xcompiler "-Wall -Wextra -Werror -fopenmp" -I./src -c src/many_simulations.cpp -o src/many_simulations.o
nvcc --use_fast_math -Xcompiler "-Wall -Wextra -Werror -fopenmp" -I./src -c src/spread_functions.cu -o src/spread_functions.o
nvcc --use_fast_math -Xcompiler "-Wall -Wextra -Werror -fopenmp" -I./src graphics/burned_probabilities_data.cpp ./src/csv.o ./src/fires.o ./src/ignition_cells.o ./src/landscape.o ./src/many_simulations.o ./src/spread_functions.o -o graphics/burned_probabilities_data
nvcc --use_fast_math -Xcompiler "-Wall -Wextra -Werror -fopenmp" -I./src graphics/fire_animation_data.cpp ./src/csv.o ./src/fires.o ./src/ignition_cells.o ./src/landscape.o ./src/many_simulations.o ./src/spread_functions.o -o graphics/fire_animation_data