#!/bin/bash
# shellcheck disable=SC2086

cur_dir="$(dirname "$(readlink -f "$0")")"
project_dir="$(realpath $cur_dir/../)"

json_dir=${project_dir}/json/train_minibatch
text_dir=${project_dir}/text/train_minibatch
data_dir=${project_dir}/dataset/gnn
work_dir=${project_dir}
py_script=train_minibatch.py
bin=${project_dir}/scripts/runner.sh

mkdir -p $json_dir
mkdir -p $text_dir

# common parameter
exp_id=0
batch_size=1024
dropout=0.3
weight_decay=0
all_lr_rates=(0.001 0.002 0.003)
all_graph_names=(pubmed reddit ogbn-arxiv ogbn-products)

function run () {
    echo "exp_id: " $exp_id
    exp_id=$((exp_id + 1))
    log_json=$json_dir/${graph_name}_${model}_${lr}.json
    log_text=$text_dir/${graph_name}_${model}_${lr}.txt
    job_name=minibatch_$exp_id

    # run for 2 days
    sbatch --partition=defq --time=11:59:00 --mem=180G --exclusive --job-name=$job_name --output=${log_text} \
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
        --num_epoch 100 \
        --batch_size $batch_size \
        --fanouts $fanouts #2>&1 | tee $log_text
}

# GAT
model=gat
for graph_name in "${all_graph_names[@]}"; do
    hid_size=128
    if [[ "$graph_name" == "pubmed" ]]; then
        hid_size=1024
        fanouts="10,10"
        num_layers=2
        num_head=4
    elif [[ "$graph_name" == "ogbn-arxiv" ]]; then
        hid_size=256
        fanouts="15,15,15,15"
        num_layers=4
        num_head=2
    elif [[ "$graph_name" == "reddit" ]]; then
        hid_size=512
        fanouts="4,4"
        num_layers=2
        num_head=2
    elif [[ "$graph_name" == "ogbn-products" ]]; then
        hid_size=128
        num_layers=2
        fanouts="4,4,4"
        num_head=2
    elif [[ "$graph_name" == "ogbn-papers100M" ]]; then
        hid_size=256
        fanouts="5,5"
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
        hid_size=256
        fanouts="10,10"
        num_layers=2
    elif [[ "$graph_name" == "ogbn-arxiv" ]]; then
        hid_size=128
        fanouts="15,15,15,15,15"
        num_layers=5
    elif [[ "$graph_name" == "reddit" ]]; then
        hid_size=128
        fanouts="4,4,4,4"
        num_layers=4
    elif [[ "$graph_name" == "ogbn-products" ]]; then
        hid_size=256
        num_layers=5
        fanouts="4,4,4,4,4"
    elif [[ "$graph_name" == "ogbn-papers100M" ]]; then
        hid_size=256
        num_layers=2    
        fanouts="5,5"
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
        hid_size=64
        fanouts="10,10,10,10,10,10"
        num_layers=6
    elif [[ "$graph_name" == "ogbn-arxiv" ]]; then
        hid_size=1024
        fanouts="15,15"
        num_layers=2
    elif [[ "$graph_name" == "reddit" ]]; then
        hid_size=512
        fanouts="4,4"
        num_layers=4
    elif [[ "$graph_name" == "ogbn-products" ]]; then
        hid_size=512
        num_layers=2
        fanouts="4,4"
    elif [[ "$graph_name" == "ogbn-papers100M" ]]; then
        hid_size=256
        num_layers=2    
        fanouts="5,5"
    fi

    for lr in "${all_lr_rates[@]}"; do
        run
    done
done