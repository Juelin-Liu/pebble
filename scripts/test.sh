#!/bin/bash
# shellcheck disable=SC2046
# shellcheck disable=SC2155

source /home/juelinliu_umass_edu/.bashrc
mamba activate quiver

LOCAL_RANK=$SLURM_LOCALID
HOST_NAME="$(hostname)"
HOSTS=$SLURM_JOB_NODELIST
HOST_IP="$(hostname -I | xargs)" # remove trailing whitespaces
MASTER_IP=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
HOST_PORT="29400"
END_POINT="$MASTER_IP:$HOST_PORT"

if [ "$LOCAL_RANK" == "0" ]; then

    echo "Hello from ${HOST_NAME}"
    echo "HOST_IP=$HOST_IP"
    echo "HOST_PORT=$HOST_PORT"
    echo "END_POINT=$END_POINT"
    echo "HOSTS=$HOSTS"
    torchrun \
        --nnodes 1 \
        --nproc-per-node 4 \
        --rdzv-backend=c10d \
        --rdzv-id="${SLURM_JOBID}" \
        --rdzv-endpoint="$END_POINT" \
        test.py

    echo "Bye from ${HOST_NAME}"

fi
