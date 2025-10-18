#!/usr/bin/env bash

set -euo pipefail

data_dir=''
if [ $# -ge 1 ]; then
    data_dir=${1}
else
    data_dir='datasets/quadrant'
fi

# bin/classify <(bin/generate_reservoir -s 250 -p 0.9 -f 20 -c 16 -o 0.3 | framework-open/bin/network_tool) "${data_dir}"/data.csv "${data_dir}"/labels.csv 0.001 11 1000000
bin/classify out.json "${data_dir}"/data.csv "${data_dir}"/labels.csv 0.025 11 50000 0.00000001 '[0, 0]' '[100, 100]' 10 4
