set -x
make clean
nvcc -O3 -Xcompiler "-Wall -Wextra -Werror -fopenmp" -I./src -c src/csv.cpp -o src/csv.o
nvcc -O3 -Xcompiler "-Wall -Wextra -Werror -fopenmp" -I./src -c src/fires.cpp -o src/fires.o
nvcc -O3 -Xcompiler "-Wall -Wextra -Werror -fopenmp" -I./src -c src/ignition_cells.cpp -o src/ignition_cells.o
nvcc -O3 -Xcompiler "-Wall -Wextra -Werror -fopenmp" -I./src -c src/landscape.cpp -o src/landscape.o
nvcc -O3 -Xcompiler "-Wall -Wextra -Werror -fopenmp" -I./src -c src/many_simulations.cpp -o src/many_simulations.o
nvcc -O3 -Xcompiler "-Wall -Wextra -Werror -fopenmp" -I./src -c src/spread_functions.cu -o src/spread_functions.o
nvcc -O3 -Xcompiler "-Wall -Wextra -Werror -fopenmp" -I./src graphics/burned_probabilities_data.cpp ./src/csv.o ./src/fires.o ./src/ignition_cells.o ./src/landscape.o ./src/many_simulations.o ./src/spread_functions.o -o graphics/burned_probabilities_data
nvcc -O3 -Xcompiler "-Wall -Wextra -Werror -fopenmp" -I./src graphics/fire_animation_data.cpp ./src/csv.o ./src/fires.o ./src/ignition_cells.o ./src/landscape.o ./src/many_simulations.o ./src/spread_functions.o -o graphics/fire_animation_data