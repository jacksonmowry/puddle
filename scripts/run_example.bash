#!/usr/bin/env bash

set -euo pipefail

data_dir=''
learning_rate=0.025
cpu_threads=$(nproc)
epochs=1000000
lambda=0.00000001
num_bins=10
input_file="best_reservoir.json"

while getopts "dr:t:e:l:b:i:" opt; do
    case ${opt} in
        r )
            learning_rate=${OPTARG}
            ;;
        t )
            cpu_threads=${OPTARG}
            ;;
        e )
            epochs=${OPTARG}
            ;;
        l )
            lambda=${OPTARG}
            ;;
        b )
            num_bins=${OPTARG}
            ;;
        i )
            input_file=${OPTARG}
            ;;
        \? )
            echo "Usage: $0 [options] <data_directory>"
            echo "Options:"
            echo "  -r <learning_rate>     Learning rate (default: 0.025)"
            echo "  -t <cpu_threads>       Number of CPU threads to use (default: number of processors)"
            echo "  -e <epochs>            Number of training epochs (default: 1000000)"
            echo "  -l <lambda>            Regularization parameter (default: 0.00000001)"
            echo "  -b <num_bins>          Number of bins (default: 10)"
            echo "  -i <input_file>        Input file (default: best_reservoir.json)"
            exit 1
            ;;
    esac
done

data_dir=${@:$OPTIND:1}

if [ -z "$data_dir" ]; then
    echo "Usage: $0 [options] <data_directory>"
    echo "Options:"
    echo "  -r <learning_rate>     Learning rate (default: 0.025)"
    echo "  -t <cpu_threads>       Number of CPU threads to use (default: number of processors)"
    echo "  -e <epochs>            Number of training epochs (default: 1000000)"
    echo "  -l <lambda>            Regularization parameter (default: 0.00000001)"
    echo "  -b <num_bins>          Number of bins (default: 10)"
    echo "  -i <input_file>        Input file (default: best_reservoir.json)"
    exit 1
fi

data_range=$(bin/data_preprocessing <${data_dir}/data.csv)
label_count=$(sort -n <${data_dir}/labels.csv | uniq | wc -l)
bin/classify $input_file \
    "${data_dir}"/data.csv \
    "${data_dir}"/labels.csv \
    ${learning_rate} \
    ${cpu_threads} \
    ${epochs} \
    ${lambda} \
    $(grep 'Min' <<<${data_range} | awk '{print $2}') \
    $(grep 'Max' <<<${data_range} | awk '{print $2}') \
    ${num_bins} "${label_count}"
