#!/usr/bin/env bash

set -euo pipefail

data_dir=''
if [ $# -gt 1 ]; then
    data_dir=${1}
else
    data_dir='datasets/quadrant'
fi

bin/classify <(bin/generate_reservoir -p 0.15 -f 20 -c 4 -o 0.2 | framework-open/bin/network_tool) "${data_dir}"/data.csv "${data_dir}"/labels.csv
