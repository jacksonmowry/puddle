#!/usr/bin/env bash

set -euo pipefail

data_dir=''
if [ $# -ge 1 ]; then
    data_dir=${1}
else
    data_dir='datasets/quadrant'
fi

learning_rate=0.025
cpu_threads=$(nproc)
epochs=1000000
lambda=0.00000001
num_bins=10

data_range=$(bin/data_preprocessing <${data_dir}/data.csv)
label_count=$(sort -n <${data_dir}/labels.csv | uniq | wc -l)
bin/classify out.json \
    "${data_dir}"/data.csv \
    "${data_dir}"/labels.csv \
    ${learning_rate} \
    ${cpu_threads} \
    ${epochs} \
    ${lambda} \
    $(grep 'Min' <<<${data_range} | awk '{print $2}') \
    $(grep 'Max' <<<${data_range} | awk '{print $2}') \
    ${num_bins} "${label_count}"
