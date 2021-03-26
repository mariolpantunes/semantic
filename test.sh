#!/usr/bin/env bash

datasets=( en-mc-30.csv en-iot-30.csv )
neighborhoods=( 3 5 7 )
nmf=( 2 3 4 )

for d in "${datasets[@]}"; do
    for n in "${neighborhoods[@]}"; do
        for k in "${nmf[@]}"; do
            python main.py -d $d -n $n -k $k
        done
    done
done