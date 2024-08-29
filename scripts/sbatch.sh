#!/bin/bash

srun --nodes=1 \
	--ntasks-per-node=4 \
	--gpus-per-node=4 \
	--partition gpu \
    --time=00:01:30 \
	${PWD}/test.sh
