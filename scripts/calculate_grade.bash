#!/usr/bin/env bash

set -euo pipefail

data_dir=''
if [ $# -gt 1 ]; then
    data_dir=${1}
else
    data_dir='datasets/quadrant'
fi

bin/reservoir_grade <(bin/generate_reservoir -s 250 -p 0.8 -f 20 -c 4 -o 0.3 | framework-open/bin/network_tool) "${data_dir}"/data.csv "${data_dir}"/labels.csv
