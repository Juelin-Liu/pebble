#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
torchrun \
    --nnodes 1 \
    --nproc-per-node 4 \
    train_dgl_dist.py \
    --num_host 1 \
    --num_gpu_per_host 4 \
    --data_dir /mnt \
    --graph_name orkut \
    --fanouts 10,10 \
    --model sage \
    --hid_size 512 \
    --log_file dgl-orkut-nd1.json \
    --num_epoch 10
