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

# all_batch_sizes=(128 256 512 1024 2048 4096 8192 16384 32768 65536)
# all_fanouts=("4,4,4" "15,15,15" "50,50,50")

all_models=(gcn gat)
all_num_layers=(2 3)
all_hid_size=(64 128 256)
all_lr=("0.00001" "0.0001" "0.001" "0.01" "0.1")
all_wd=("0.00001" "0.0001" "0.001" "0.01" "0.1")
all_dropout=(0.3 0.5 0.7 0.9)
all_graph_names=(pubmed reddit ogbn-proteins ogbn-arxiv ogbn-products ogbn-papers100M)

# all_models=(gcn gat)
# all_num_layers=(2)
# all_hid_size=(64)
# all_lr=("0.001")
# all_wd=("0.001")
# all_dropout=(0.5)
# all_graph_names=(pubmed reddit ogbn-proteins ogbn-arxiv ogbn-products ogbn-papers100M)

exp_id=0
mkdir -p $json_dir
mkdir -p $text_dir

for model in "${all_models[@]}"; do
    if [[ "$model" == "gcn" ]]; then
        all_num_head=(0)
    elif [[ "$model" == "gat" ]]; then
        all_num_head=(4 8 12)
    fi

    for num_head in "${all_num_head[@]}"; do
        for num_layers in "${all_num_layers[@]}"; do
            for hid_size in "${all_hid_size[@]}"; do
                for lr in "${all_lr[@]}"; do
                    for weight_decay in "${all_wd[@]}"; do
                        for dropout in "${all_dropout[@]}"; do
                            for graph_name in "${all_graph_names[@]}"; do
                                echo "exp_id: " $exp_id
                                exp_id=$((exp_id + 1))
                                log_json=$json_dir/$exp_id.json
                                log_text=$text_dir/$exp_id.txt
                                job_name=full_$exp_id

                                sbatch --partition=defq --time=00:20:00 --mem=180G --exclusive --job-name=$job_name --output=${log_text} \
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
                                    --num_epoch 300 \
                                    --batch_size -1 \
                                    --fanouts -1 #2>&1 | tee $log_text
                            done
                        done
                    done
                done
            done
        done
    done
done
