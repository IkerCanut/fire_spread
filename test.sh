for i in $(seq 1 40);
do
    for l in $(cat a);
    do
        ./graphics/fire_animation_data ./data/$l > /dev/null;
    done
done
