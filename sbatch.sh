#!/bin/bash

work_dir="$(dirname "$(readlink -f "$0")")"
work_dir=$(realpath "$work_dir")

num_gpu_per_host=4
gpu_model=a100

for num_host in 1 2 3; do
    sbatch \
    --partition=gpu-preempt \
        --nodes=$num_host \
        --ntasks-per-node=1 \
        --gpus-per-node=${gpu_model}:$num_gpu_per_host \
        --cpus-per-gpu=4 \
        --mem=489GB \
        --time=03:01:00 \
    srun.sh \
        --work_dir "$work_dir" \
        --num_host $num_host \
        --num_gpu_per_host $num_gpu_per_host \
        --gpu_model $gpu_model
done
