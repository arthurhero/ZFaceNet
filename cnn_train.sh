#!/bin/bash

for value in $(seq 5)
do
    echo $value
    python train.py
done
