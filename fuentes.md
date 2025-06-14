# NOTAS

Pasamos la funcion que checkea errores llamada CUDA_CHECK vista en:
https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api

En spread_probability_device usamos las funciones matematicas de CUDA:
sinf, cosf, expf

# FUNCIONES Y MACROS

## CUDA_CHECK
cudaError_t: tipo de error devuelto por funciones de la Runtime API de CUDA.
cudaGetErrorString(e): convierte un código de error CUDA en un mensaje legible.
Esta macro se usa en el host (CPU) para comprobar errores de llamadas CUDA.

## __global__ void setup_rand_states
__global__ : indica que es un kernel de CUDA, ie. es una funcion que se ejecuta en la GPU y es llamada dsde el host.

blockIdx.x, blockDim.x, threadIdx.x: identificadores propios del modelo de programación CUDA. Usados para calcular un índice de hilo único.

curandState: estructura de estado del generador aleatorio de CUDA (cuRAND).

curand_init(...): inicializa el generador aleatorio para ese hilo en GPU.

Init de un generador aleatorio independiente por hilo.

## __device__ float spread_probability_device

__device__ : Indica que solo puede ser llamada desde código que también se ejecuta en el device (GPU). Se ejecuta en la GPU, no en la CPU.


## __global__ void evaluate_spread_kernel()

Identificacion del hilo CUDA
unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

Acceso paralelo seguro usando operaciones atomicas como
atomicAdd: suma atómica; varios hilos pueden incrementar un valor sin interferirse.
atomicCAS: "Compare And Swap"; cambia un valor solo si tiene cierto contenido esperado.
atomicExch: intercambio atómico de valores.

Memoria de hilos: uso de curandState
curandState* rand_state = &d_rand_states[thread_idx];
Cada hilo tiene su propio estado de generador de números aleatorios (curandState), lo cual permite generar aleatoriedad paralela sin conflictos.

Tipos de punteros en GPU
Todos los punteros (como d_cells, d_burned_bin, etc.) apuntan a memoria en la GPU (device memory).
El prefijo d_ es una convención para indicar que están en el device.

Lógica condicional por hilo
if (thread_idx >= work_size) return;
Cada hilo decide si tiene trabajo que hacer.
En CUDA, no todos los hilos hacen el mismo trabajo, y esta línea asegura que hilos sobrantes no hagan nada.

