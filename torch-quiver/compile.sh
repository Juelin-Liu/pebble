#!/bin/bash
export TORCH_CUDA_ARCH_LIST="7.0 8.0 8.6 8.9 9.0"
MAX_JOBS=32 QUIVER_ENABLE_CUDA=1 python setup.py install
