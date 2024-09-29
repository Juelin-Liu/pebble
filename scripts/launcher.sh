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

source /home/juelinliu_umass_edu/.profile
conda activate pebble

LOCAL_RANK=$SLURM_LOCALID
HOST_NAME="$(hostname)"
HOSTS=$SLURM_JOB_NODELIST
HOST_IP="$(hostname -I | xargs)" # remove trailing whitespaces
MASTER_NAME=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_IP=$(scontrol show hostname "$MASTER_NAME" | xargs -I{} getent hosts {} | awk '{print $1}')
MASTER_PORT="29400"

if [ "$MASTER_NAME" == "$HOST_NAME" ]; then
    MASTER_IP=$("$HOST_IP" | awk '{print $1}')
fi
END_POINT="$MASTER_IP:$MASTER_PORT"
PYTHON=$(which python)

if [ "$LOCAL_RANK" == "0" ]; then


    echo "Hello from ${HOST_NAME} (${HOST_IP})"
    echo "NUM_NODES=${SLURM_JOB_NUM_NODES}"
    echo "END_POINT=$END_POINT"
    echo "HOSTS=$HOSTS"
    echo "HOST_IP=$HOST_IP"
    echo "PYTHON=$PYTHON"
    
    cd "$work_dir" || exit
    new_data_dir=/tmp
    cp -r "${data_dir}"/"${graph_name}" $new_data_dir &
    sleep 150s
    wait

    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
    torchrun \
        --nnodes ${SLURM_JOB_NUM_NODES} \
        --nproc-per-node ${num_gpu_per_host} \
        --rdzv-backend=c10d \
        --rdzv-id="${SLURM_JOBID}" \
        --rdzv-endpoint="$END_POINT" \
        $py_script \
            --log_file $log_file \
            --batch_size $batch_size \
            --fanouts $fanouts \
            --num_epoch $num_epoch \
            --hid_size $hid_size \
            --num_head $num_head \
            --lr $lr \
            --weight_decay $weight_decay \
            --dropout $dropout \
            --num_host $num_host \
            --num_gpu_per_host $num_gpu_per_host \
            --data_dir $new_data_dir \
            --graph_name $graph_name \
            --model $model

    echo "Bye from ${HOST_NAME}"
fi
