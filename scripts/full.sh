#!/bin/bash
# shellcheck disable=SC2086

cur_dir="$(dirname "$(readlink -f "$0")")"
project_dir="$(realpath $cur_dir/../)"

json_dir=${project_dir}/json/train_full
text_dir=${project_dir}/text/train_full
data_dir=${project_dir}/dataset/gnn
work_dir=${project_dir}
py_script=train_full.py
bin=${project_dir}/scripts/runner.sh

mkdir -p $json_dir
mkdir -p $text_dir

# common parameter
exp_id=0
batch_size=1024
dropout=0.3
weight_decay=0
fanouts="-1"
all_lr_rates=(0.001 0.002 0.003)
all_graph_names=(pubmed reddit ogbn-arxiv ogbn-products)

function run () {
    echo "exp_id: " $exp_id
    exp_id=$((exp_id + 1))
    log_json=$json_dir/$exp_id.json
    log_text=$text_dir/$exp_id.txt
    # job_name=minibatch_$exp_id
    # sbatch --partition=defq --time=00:20:00 --mem=180G --exclusive --job-name=$job_name --output=${log_text} \
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
        --world_size 1 \
        --num_partition 1 \
        --data_dir $data_dir \
        --graph_name $graph_name \
        --num_epoch 500 \
        --batch_size $batch_size \
        --fanouts $fanouts 2>&1 | tee $log_text
}

# GAT
model=gat
for graph_name in "${all_graph_names[@]}"; do
    hid_size=128
    if [[ "$graph_name" == "pubmed" ]]; then
        hid_size=1024
        num_layers=3
        num_head=1
    elif [[ "$graph_name" == "ogbn-arxiv" ]]; then
        hid_size=1024
        num_layers=3
        num_head=2
    elif [[ "$graph_name" == "reddit" ]]; then
        hid_size=1024
        num_layers=2
        num_head=2
    elif [[ "$graph_name" == "ogbn-products" ]]; then
        hid_size=256
        num_layers=2
        num_head=2
    fi

    for lr in "${all_lr_rates[@]}"; do
        run
    done
done

# GraphSage
num_head=0
model=sage
for graph_name in "${all_graph_names[@]}"; do
    hid_size=128
    if [[ "$graph_name" == "pubmed" ]]; then
        hid_size=64
        num_layers=5
    elif [[ "$graph_name" == "ogbn-arxiv" ]]; then
        hid_size=512
        num_layers=3
    elif [[ "$graph_name" == "reddit" ]]; then
        hid_size=1024
        num_layers=4
    elif [[ "$graph_name" == "ogbn-products" ]]; then
        hid_size=512
        num_layers=5
    fi

    for lr in "${all_lr_rates[@]}"; do
        run
    done
done

# GCN
model=gcn
for graph_name in "${all_graph_names[@]}"; do
    hid_size=128
    if [[ "$graph_name" == "pubmed" ]]; then
        hid_size=512
        num_layers=2
    elif [[ "$graph_name" == "ogbn-arxiv" ]]; then
        hid_size=1024
        num_layers=2
    elif [[ "$graph_name" == "reddit" ]]; then
        hid_size=1024
        num_layers=2
    elif [[ "$graph_name" == "ogbn-products" ]]; then
        hid_size=512
        num_layers=3
    fi

    for lr in "${all_lr_rates[@]}"; do
        run
    done
done