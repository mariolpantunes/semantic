#!/usr/bin/env bash

datasets=( en-mc-30.csv en-iot-30.csv )
cutoff=( pareto20 pareto80 ) 
neighborhoods=( 3 5 7 )
nmf=( 1 2 3 4 5 )

for d in "${datasets[@]}"; do
    for c in "${cutoff[@]}"; do
        for n in "${neighborhoods[@]}"; do
            for k in "${nmf[@]}"; do
                python main.py -d $d -c $c -n $n -k $k
            done
        done
    done
done