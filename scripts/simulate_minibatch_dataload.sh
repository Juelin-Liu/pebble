#!/bin/bash

# shellcheck disable=SC2086

cur_dir="$(dirname "$(readlink -f "$0")")"
project_dir="$(realpath $cur_dir/../)"

json_dir=${project_dir}/json/simulate_minibatch_dataload
text_dir=${project_dir}/text/simulate_minibatch_dataload
data_dir=${project_dir}/dataset/gnn
work_dir=${project_dir}
py_script=simulate_minibatch_dataload.py
bin=${project_dir}/scripts/runner.sh

# TODO: implement this script with correct hyper parameters