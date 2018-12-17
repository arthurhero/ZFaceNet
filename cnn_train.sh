#!/bin/bash

for value in $(seq 200)
do
    echo $value
    python train.py
done
