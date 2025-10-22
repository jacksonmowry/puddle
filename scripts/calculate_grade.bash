#!/usr/bin/env bash

set -euo pipefail

data_dir=''
s=250
p=0.025
f=-1
c=64
o=0.3
r=$(od -An -N4 -tu4 < /dev/urandom)
num_bins=10

while getopts "ds:p:f:c:o:r:b:" opt; do
    case ${opt} in
        s )
            s=${OPTARG}
            ;;
        p )
            p=${OPTARG}
            ;;
        f )
            f=${OPTARG}
            ;;
        c )
            c=${OPTARG}
            ;;
        o )
            o=${OPTARG}
            ;;
        r )
            r=${OPTARG}
            ;;
        b )
            num_bins=${OPTARG}
            ;;
        \? )
            echo "Usage: $0 [options] <data_directory>"
            echo "Options:"
            echo "  -s <size>                Reservoir size (default: 250)"
            echo "  -p <connection_probability>  Connection probability (default: 0.025)"
            echo "  -f <feature_neurons>     Number of feature neurons (default: calculated)"
            echo "  -c <class_neurons>       Number of class neurons (default: 64)"
            echo "  -o <output_weight>       Output weight (default: 0.3)"
            echo "  -r <seed>                Random seed (default: 0)"
            echo "  -b <num_bins>            Number of bins (default: 10)"
            exit 1
            ;;
    esac
done

data_dir=${@:$OPTIND:1}

if [ -z "$data_dir" ]; then
    echo "Usage: $0 [options] <data_directory>"
    echo "Options:"
    echo "  -s <size>                Reservoir size (default: 250)"
    echo "  -p <connection_probability>  Connection probability (default: 0.025)"
    echo "  -f <feature_neurons>     Number of feature neurons (default: calculated)"
    echo "  -c <class_neurons>       Number of class neurons (default: 64)"
    echo "  -o <output_weight>       Output weight (default: 0.3)"
    echo "  -r <seed>                Random seed (default: 0)"
    echo "  -b <num_bins>            Number of bins (default: 10)"
    exit 1
fi

cpu_threads=$(nproc)
data_range=$(bin/data_preprocessing <${data_dir}/data.csv)
num_features=$(grep 'Num' <<<${data_range} | awk '{print $2}')
label_count=$(sort -n <${data_dir}/labels.csv | uniq | wc -l)

if [ ${f} -eq -1 ]; then
    f=$((num_bins * num_features))
fi

echo ${s} ${p} ${f} ${c} ${o} ${r} ${num_bins} ${data_dir}

bin/generate_reservoir \
    -s ${s} \
    -p ${p} \
    -f ${f} \
    -c ${c} \
    -o ${o} \
    -r ${r} | framework-open/bin/network_tool >out.json

bin/grade \
    out.json \
    "${data_dir}"/data.csv \
    "${data_dir}"/labels.csv \
    ${cpu_threads} \
    $(grep 'Min' <<<${data_range} | awk '{print $2}') \
    $(grep 'Max' <<<${data_range} | awk '{print $2}') \
    ${num_bins} \
    ${label_count}
