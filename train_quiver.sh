#!/bin/bash
# shellcheck disable=SC2046
# shellcheck disable=SC2155


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
        --num_host) num_host="$2"; shift ;;
        --num_gpu_per_host) num_gpu_per_host="$2"; shift ;;
        --data_dir) data_dir="$2"; shift ;;
        --graph_name) graph_name="$2"; shift ;;
        --model) model="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

eval "$(conda shell.bash hook)"
conda activate quiver

LOCAL_RANK=$SLURM_LOCALID
HOST_NAME="$(hostname)"
HOSTS=$SLURM_JOB_NODELIST
HOST_IP="$(hostname -I | xargs)" # remove trailing whitespaces
MASTER_NAME=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="29400"
END_POINT="$MASTER_NAME:$MASTER_PORT"
PYTHON=$(which python)

if [ "$LOCAL_RANK" == "0" ]; then

    echo "Hello from ${HOST_NAME} (${HOST_IP})"
    echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
    echo "END_POINT=$END_POINT"
    echo "HOSTS=$HOSTS"
    echo "PYTHON=$PYTHON"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

    cd $work_dir

    torchrun \
        --nnodes ${SLURM_JOB_NUM_NODES} \
        --nproc-per-node 1 \
        --rdzv-backend=c10d \
        --rdzv-id="${SLURM_JOBID}" \
        --rdzv-endpoint="$END_POINT" \
        $py_script \
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
            --num_host $num_host \
            --num_gpu_per_host $num_gpu_per_host \
            --data_dir $data_dir \
            --graph_name $graph_name \
            --model $model

    # if [ -f $log_file ] && [[ $log_file == *".json" ]] && [[ ${HOST_NAME} == "${MASTER_NAME}" ]]; then
    #     echo "formatting output json:" $log_file
    #     python3 -m json.tool $log_file > $log_file
    # fi

    echo "Bye from ${HOST_NAME}"
fi
