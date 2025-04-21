#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Uso: $0 archivo_entrada"
    exit 1
fi

archivo="$1"

datos=()
while IFS= read -r linea; do
    linea_cleaned=$(echo "$linea" | sed 's/Celdas\/ms: //')
    datos+=("$linea_cleaned")
done < "$archivo"

num_medidas=11
num_experimentos=$(( ${#datos[@]} / num_medidas ))

for ((i=0; i<num_medidas; i++)); do
    for ((j=0; j<num_experimentos; j++)); do
        index=$(( j * num_medidas + i ))
        printf "%s \t " "${datos[index]}"
    done
    echo ""
done