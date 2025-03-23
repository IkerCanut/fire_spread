Podemos medir el rendimiento como:

`Rendimiento = Celdas / Tiempo`

Y podemos calcular las celdas con el metadata de cada problema.

Ademas calculamos el tiempo del bucle principal usando 
```
int t = omp_get_wtime();
while (burning_size > 0) {
    [SNIP]
}
std::cerr << "Tiempo: " << omp_get_wtime() - t << std::endl;
```

Tenemos los datos de cells en el spreadsheet.

+ Es escalable para diferentes tamaños de simulacion.
+ Mayor valor indica mejor rendimiento.
+ Facil de medir y comparar.

-------------------------------------------------------------

Otra forma podria ser medir las floating point operations per second:

`Rendimiento2 = FLOPS / IPS`

FLOPS (Floating Point Operations per Second): Cuantas operaciones de punto flotante se realizan por segundo.

IPS (Instructions per Second): Cuantas instrucciones se ejecutan por segundo.

+ Si el valor es alto, la mayoria de las instrucciones son utiles.
+ Si el valor es bajo, muchas instucciones no contribuyen al calculo (acceso a memoria, saltos, ...). El codigo podria estar limitado por accesos a memoria en lugar de computo puro.