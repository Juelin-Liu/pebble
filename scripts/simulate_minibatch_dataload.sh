#!/bin/bash

# shellcheck disable=SC2086
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

cur_dir="$(dirname "$(readlink -f "$0")")"
project_dir="$(realpath $cur_dir/../)"

json_dir=${project_dir}/json/simulate_minibatch_dataload
text_dir=${project_dir}/text/simulate_minibatch_dataload
data_dir=${project_dir}/dataset/gnn
py_script=${project_dir}/simulate_minibatch_dataload.py


# tested graphs
all_graph_names=(pubmed reddit ogbn-arxiv ogbn-products)

# common hyperparameters
num_epoch=10
batch_size=1024
num_partition=4

for graph_name in "${all_graph_names[@]}"; do
    if [[ "$graph_name" == "pubmed" ]]; then
        hid_size=256
        fanouts=("5,5" "10,10" "15,15" "20,20")
    elif [[ "$graph_name" == "reddit" ]]; then
        hid_size=512
        fanouts=("5,5,5,5" "10,10,10,10" "15,15,15,15" "20,20,20,20")
    elif [[ "$graph_name" == "ogbn-arxiv" ]]; then
        hid_size=128
        fanouts=("5,5,5,5,5" "10,10,10,10,10" "15,15,15,15,15" "20,20,20,20,20")
    elif [[ "$graph_name" == "ogbn-products" ]]; then
        hid_size=256
        fanouts=("5,5,5,5,5" "10,10,10,10,10" "15,15,15,15,15" "20,20,20,20,20")
    fi

    exp_id=0
    for fanout in "${fanouts[@]}"; do
        exp_id=$((exp_id + 1))
        log_json=$json_dir/$graph_name/$exp_id.json
        log_text=$text_dir/$graph_name/$exp_id.txt

        mkdir -p $text_dir/$graph_name/
        mkdir -p $json_dir/$graph_name/
        python3 ${py_script} \
            --log_file $log_json \
            --fanouts $fanout \
            --graph_name $graph_name \
            --num_epoch ${num_epoch} \
            --batch_size ${batch_size} \
            --num_partition ${num_partition} \
            --data_dir ${data_dir} \
            --hid_size ${hid_size} 2>&1 | tee $log_text
    done
done
