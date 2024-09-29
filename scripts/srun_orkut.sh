#!/bin/bash
# shellcheck disable=SC2086

while [[ "$#" -gt 0 ]]; do
    case $1 in
    --work_dir)
        work_dir="$2"
        shift
        ;;
    --gpu_model)
        gpu_model="$2"
        shift
        ;;
    --num_host)
        num_host="$2"
        shift
        ;;
    --system)
        system="$2"
        shift
        ;;
    --num_gpu_per_host)
        num_gpu_per_host="$2"
        shift
        ;;
    *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
    shift
done

project_dir="$(realpath $work_dir)"
json_dir=${project_dir}/json/${system}
text_dir=${project_dir}/text/${system}
data_dir=${project_dir}/dataset
py_script=${project_dir}/train_${system}.py
bin=${project_dir}/launcher.sh

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
num_epoch=10
graph_name=orkut
fanouts="10,10,10"
memory=495GB

log_json=$json_dir/${graph_name}_n${num_host}_${gpu_model}.json
log_text=$text_dir/${graph_name}_n${num_host}_${gpu_model}.txt
job_name=${system}_${graph_name}_n${num_host}_${gpu_model}

srun --partition=gpu-preempt \
    --ntasks-per-node=1 \
    --gpus-per-node=${gpu_model}:${num_gpu_per_host} \
    --cpus-per-gpu=4 \
    --mem=${memory} \
    --time=01:00:00 \
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
    --lr $lr \
    --weight_decay $weight_decay \
    --dropout $dropout \
    --num_host ${num_host} \
    --num_gpu_per_host $num_gpu_per_host \
    --data_dir $data_dir \
    --graph_name $graph_name \
    --num_epoch $num_epoch \
    --batch_size $batch_size \
    --fanouts $fanouts
