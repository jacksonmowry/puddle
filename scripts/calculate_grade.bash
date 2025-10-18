#!/usr/bin/env bash

set -euo pipefail

data_dir=''
if [ $# -ge 1 ]; then
    data_dir=${1}
else
    data_dir='datasets/quadrant'
fi

cpu_threads=$(nproc)
num_bins=10
data_range=$(bin/data_preprocessing <${data_dir}/data.csv)
num_features=$(grep 'Num' <<<${data_range} | awk '{print $2}')
label_count=$(sort -n <${data_dir}/labels.csv | uniq | wc -l)

bin/generate_reservoir \
    -s 250 \
    -p 0.025 \
    -f $((num_bins * num_features)) \
    -c 64 \
    -o 0.3 | framework-open/bin/network_tool >out.json

bin/grade \
    out.json \
    "${data_dir}"/data.csv \
    "${data_dir}"/labels.csv \
    ${cpu_threads} \
    $(grep 'Min' <<<${data_range} | awk '{print $2}') \
    $(grep 'Max' <<<${data_range} | awk '{print $2}') \
    ${num_bins} \
    ${label_count}
