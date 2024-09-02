#!/bin/bash
# shellcheck disable=SC2086


# function launch_quiver() {
#     log_json=$json_dir/${graph_name}_n$num_host.json
#     log_text=$text_dir/${graph_name}_n$num_host.txt
#     job_name=quiver_${graph_name}_n$num_host

#     srun --partition=gpu \
#         --ntasks-per-node=1 \
#         --gpus-per-node=a100:${num_gpu_per_host} \
#         --cpus-per-gpu=4 \
#         --mem=${memory} \
#         --time=00:30:00 \
#         --nodes=$num_host \
#         --job-name=$job_name \
#         --output=$log_text \
#         $bin \
#         --work_dir $work_dir \
#         --py_script $py_script \
#         --log_file $log_json \
#         --hid_size $hid_size \
#         --model $model \
#         --num_head $num_head \
#         --num_layers $num_layers \
#         --lr $lr \
#         --weight_decay $weight_decay \
#         --dropout $dropout \
#         --num_host ${num_host} \
#         --num_gpu_per_host $num_gpu_per_host \
#         --data_dir $data_dir \
#         --graph_name $graph_name \
#         --num_epoch 50 \
#         --batch_size $batch_size \
#         --fanouts $fanouts #2>&1 | tee $log_text
# }

work_dir="$(dirname "$(readlink -f "$0")")"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --work_dir) work_dir="$2"; shift ;;
        --num_host) num_host="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

project_dir="$(realpath $work_dir)"
json_dir=${project_dir}/json/quiver
text_dir=${project_dir}/text/quiver
data_dir=${project_dir}/dataset/gnn
py_script=${project_dir}/train_quiver.py
bin=${project_dir}/train_quiver.sh

mkdir -p $json_dir
mkdir -p $text_dir

# common parameter
batch_size=1024
dropout=0.3
weight_decay=0.005
lr=0.001
hid_size=512
model=sage
num_head=0
num_gpu_per_host=4

# graph specific parameter
graph_name=orkut
fanouts="10,10,10"
num_layers=3
memory=123GB

for num_host in 1 2 3; do
    log_json=$json_dir/${graph_name}_n$num_host.json
    log_text=$text_dir/${graph_name}_n$num_host.txt
    job_name=quiver_${graph_name}_n$num_host

    srun --partition=gpu \
        --ntasks-per-node=1 \
        --gpus-per-node=a100:${num_gpu_per_host} \
        --cpus-per-gpu=4 \
        --mem=${memory} \
        --time=00:30:00 \
        --nodes=$num_host \
        --job-name=$job_name \
        --output=$log_text \
        $bin \
        --work_dir $work_dir \
        --py_script $py_script \
        --log_file $log_json \
        --hid_size $hid_size \
        --model $model \
        --num_head $num_head \
        --num_layers $num_layers \
        --lr $lr \
        --weight_decay $weight_decay \
        --dropout $dropout \
        --num_host ${num_host} \
        --num_gpu_per_host $num_gpu_per_host \
        --data_dir $data_dir \
        --graph_name $graph_name \
        --num_epoch 50 \
        --batch_size $batch_size \
        --fanouts $fanouts
done

graph_name=ogbn-papers100M
fanouts="10,10"
num_layers=2
memory=488GB
for num_host in 1 2 3; do
    log_json=$json_dir/${graph_name}_n$num_host.json
    log_text=$text_dir/${graph_name}_n$num_host.txt
    job_name=quiver_${graph_name}_n$num_host

    srun --partition=gpu \
        --ntasks-per-node=1 \
        --gpus-per-node=a100:${num_gpu_per_host} \
        --cpus-per-gpu=4 \
        --mem=${memory} \
        --time=00:30:00 \
        --nodes=$num_host \
        --job-name=$job_name \
        --output=$log_text \
        $bin \
        --work_dir $work_dir \
        --py_script $py_script \
        --log_file $log_json \
        --hid_size $hid_size \
        --model $model \
        --num_head $num_head \
        --num_layers $num_layers \
        --lr $lr \
        --weight_decay $weight_decay \
        --dropout $dropout \
        --num_host ${num_host} \
        --num_gpu_per_host $num_gpu_per_host \
        --data_dir $data_dir \
        --graph_name $graph_name \
        --num_epoch 50 \
        --batch_size $batch_size \
        --fanouts $fanouts
done
