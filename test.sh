for i in $(seq 1 100);
do
    for l in $(cat a);
    do
        ./graphics/fire_animation_data ./data/$l > /dev/null;
    done
done