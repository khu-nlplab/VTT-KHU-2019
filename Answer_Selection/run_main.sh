#! /bin/bash

python main.py \
    --data_dir=./data \
    --mode=test \
    --use_dicts=True \
    --glove_file='glove.6B.300d.txt' \
    --is_load=True \
    --model_dir=./model \
    --dataset_type='multiple'
