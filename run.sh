#!/bin/bash

data=mnist
noise_type=flip
noise_rate=0.45

python3 main.py --dataset $data  --noise_type  $noise_type --noise_rate $noise_rate --seed 0 --lam 0.0001 --device 0 --save_dir temp