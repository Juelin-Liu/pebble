#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
torchrun \
    --nnodes 1 \
    --nproc-per-node 1 \
    train_quiver.py \
    --num_host 1 \
    --num_gpu_per_host 4 \
    --data_dir dataset/gnn/ \
    --graph_name ogbn-papers100M \
    --model sage \
    --num_epoch 10
