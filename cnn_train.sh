#!/bin/bash

for value in $(seq 10)
do
    echo $value
    python train.py
done
