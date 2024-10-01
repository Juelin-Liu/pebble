#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

work_dir="$(dirname "$(readlink -f "$0")")"
work_dir=$(realpath "$work_dir/..")

py_script=$work_dir/train_quiver_replicate.py
num_epoch=10
system_name=quiver-replicate
num_host=1
num_proc_per_node=4
num_gpu_per_node=4
log_dir=${work_dir}/results
JOBID=${system_name}
# HOST_IP="$(hostname -I | xargs)" # remove trailing whitespaces
# HOST_IP=$("$HOST_IP" | awk '{print $1}')
HOST_IP=10.0.0.5
PORT=29400
END_POINT="$HOST_IP:$PORT"
mkdir -p $log_dir
# # orkut
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
    --log_file $log_dir/${system_name}-orkut-h512-n${num_host}.json \
    --num_epoch $num_epoch

num_epoch=50
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
    --log_file $log_dir/${system_name}-paper-h512-n${num_host}.json \
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
    --log_file $log_dir/${system_name}-paper-h128-n${num_host}.json \
    --num_epoch $num_epoch
