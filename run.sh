#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TORCH_DISTRIBUTED_DEBUG=DETAIL
torchrun \
    --nnodes 1 \
    --nproc-per-node 2 \
    train_quiver_dist.py \
    --num_host 1 \
    --num_gpu_per_host 2 \
    --data_dir /tmp/dataset/gnn/ \
    --graph_name orkut \
    --fanouts 10,10 \
    --model sage \
    --num_epoch 10
