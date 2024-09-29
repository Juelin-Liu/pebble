#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

py_script=train_quiver_replicate.py
num_epoch=10
system_name=quiver-replicate
num_host=1
num_proc_per_node=1
num_gpu_per_node=4

JOBID=${system_name}
END_POINT="10.0.0.4:29400"

# orkut
torchrun \
    --nnodes ${num_host} \
    --nproc-per-node ${num_proc_per_node} \
    --rdzv-backend=c10d \
    --rdzv-id="${JOBID}" \
    --rdzv-endpoint="$END_POINT" \
    $py_script \
    --num_host ${num_host} \
    --num_gpu_per_host ${num_gpu_per_node} \
    --data_dir /mnt \
    --graph_name orkut \
    --fanouts 10,10,10 \
    --model sage \
    --hid_size 512 \
    --log_file results/${system_name}-orkut-h512-n${num_host}.json \
    --num_epoch $num_epoch

# ogbn-papers100M
torchrun \
    --nnodes ${num_host} \
    --nproc-per-node ${num_proc_per_node} \
    --rdzv-backend=c10d \
    --rdzv-id="${JOBID}" \
    --rdzv-endpoint="$END_POINT" \
    $py_script \
    --num_host ${num_host} \
    --num_gpu_per_host ${num_gpu_per_node} \
    --data_dir /mnt \
    --graph_name ogbn-papers100M \
    --fanouts 10,10 \
    --model sage \
    --hid_size 512 \
    --log_file results/${system_name}-paper-h512-n${num_host}.json \
    --num_epoch $num_epoch

torchrun \
    --nnodes ${num_host} \
    --nproc-per-node ${num_proc_per_node} \
    --rdzv-backend=c10d \
    --rdzv-id="${JOBID}" \
    --rdzv-endpoint="$END_POINT" \
    $py_script \
    --num_host ${num_host} \
    --num_gpu_per_host ${num_gpu_per_node} \
    --data_dir /mnt \
    --graph_name ogbn-papers100M \
    --fanouts 10,10 \
    --model sage \
    --hid_size 128 \
    --log_file results/${system_name}-paper-h128-n${num_host}.json \
    --num_epoch $num_epoch
