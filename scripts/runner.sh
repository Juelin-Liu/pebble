#!/bin/bash

# shellcheck disable=SC2086

source ~/.bashrc
conda activate dgl

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --work_dir) work_dir="$2"; shift ;;
        --py_script) py_script="$2"; shift ;;
        --log_file) log_file="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --fanouts) fanouts="$2"; shift ;;
        --num_epoch) num_epoch="$2"; shift ;;
        --hid_size) hid_size="$2"; shift ;;
        --num_layers) num_layers="$2"; shift ;;
        --num_head) num_head="$2"; shift ;;
        --lr) lr="$2"; shift ;;
        --weight_decay) weight_decay="$2"; shift ;;
        --dropout) dropout="$2"; shift ;;
        --world_size) world_size="$2"; shift ;;
        --num_partition) num_partition="$2"; shift ;;
        --data_dir) data_dir="$2"; shift ;;
        --graph_name) graph_name="$2"; shift ;;
        --model) model="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

cd $work_dir || exit

python3 $py_script \
    --log_file $log_file \
    --batch_size $batch_size \
    --fanouts $fanouts \
    --num_epoch $num_epoch \
    --hid_size $hid_size \
    --num_layers $num_layers \
    --num_head $num_head \
    --lr $lr \
    --weight_decay $weight_decay \
    --dropout $dropout \
    --world_size $world_size \
    --num_partition $num_partition \
    --data_dir $data_dir \
    --graph_name $graph_name \
    --model $model

if [ -f $log_file ]; then
    if [[ $log_file == *".json"]]; then
        echo "formatting output json:" $log_file
        python3 -m json.tool $log_file > $log_file
    fi
fi
