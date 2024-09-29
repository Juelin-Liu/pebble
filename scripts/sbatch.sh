#!/bin/bash

work_dir="$(dirname "$(readlink -f "$0")")"
work_dir=$(realpath "$work_dir/..")

num_gpu_per_host=4
gpu_model=a100
# system=torch_replicate

for system in "torch_replicate" "dgl_dist"; do
    for num_host in 1 2 3; do
        sbatch \
            --partition=gpu-preempt \
            --nodes=$num_host \
            --tasks-per-node=1 \
            --gpus-per-node=${gpu_model}:$num_gpu_per_host \
            --cpus-per-task=16 \
            --mem=499GB \
            --time=01:05:00 \
            srun_orkut.sh \
            --work_dir "$work_dir" \
            --num_host $num_host \
            --num_gpu_per_host $num_gpu_per_host \
            --gpu_model $gpu_model \
            --system $system

        sbatch \
            --partition=gpu-preempt \
            --nodes=$num_host \
            --tasks-per-node=1 \
            --gpus-per-node=${gpu_model}:$num_gpu_per_host \
            --cpus-per-task=16 \
            --mem=496GB \
            --time=01:05:00 \
            srun_paper.sh \
            --work_dir "$work_dir" \
            --num_host $num_host \
            --num_gpu_per_host $num_gpu_per_host \
            --gpu_model $gpu_model \
            --system $system
    done
done
