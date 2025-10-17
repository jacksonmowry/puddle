#!/usr/bin/env bash

set -euo pipefail

data_dir=''
if [ $# -ge 1 ]; then
    data_dir=${1}
else
    data_dir='datasets/quadrant'
fi

bin/classify out.json "${data_dir}"/data.csv "${data_dir}"/labels.csv 0 0.001 10
