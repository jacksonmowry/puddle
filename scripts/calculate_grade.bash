#!/usr/bin/env bash

set -euo pipefail

data_dir=''
if [ $# -ge 1 ]; then
    data_dir=${1}
else
    data_dir='datasets/quadrant'
fi

bin/generate_reservoir -s 250 -p 0.025 -f 20 -c 64 -o 0.3 | framework-open/bin/network_tool >out.json
bin/grade out.json "${data_dir}"/data.csv "${data_dir}"/labels.csv 10
