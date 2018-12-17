#!/bin/bash

for value in $(seq 20)
do
    echo $value
    python train.py
done
