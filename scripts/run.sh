#!/bin/bash

#SBATCH -p <dummy_name>
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:Tesla-V100-32GB:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=256G
#SBATCH --job-name=quiver_distributed
#SBATCH --output=slurm.out

source ~/.bashrc

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False


torchrun --nnodes 1 --nproc-per-node 1 --standalone \
    train_quiver.py \
    --graph_name ogbn-products \
    --data_dir ~/work2/dataset/gnn/ \
    --num_host 1 \
    --num_gpu_per_host 4 \
    --num_epoch 5 >out.log 2>&1