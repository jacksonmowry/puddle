#!/usr/bin/env bash

set -euo pipefail

data_dir=''
s="250"
p="0.025"
c="64"
o="0.3"
N=10
num_bins="-b 10"
output_file="best_reservoir.json"

while getopts "s:p:f:c:o:b:n:w:" opt; do
    case ${opt} in
    s)
        s="${OPTARG}"
        ;;
    p)
        p="${OPTARG}"
        ;;
    c)
        c="${OPTARG}"
        ;;
    o)
        o="${OPTARG}"
        ;;
    b)
        num_bins="${OPTARG}"
        ;;
    n)
        N=${OPTARG}
        ;;
    w)
        output_file="${OPTARG}"
        ;;
    \?)
        echo "Usage: $0 [options] <data_directory>"
        echo "Options:"
        echo "  -s <size>                    Reservoir size"
        echo "  -p <connection_probability>  Connection probability"
        echo "  -f <feature_neurons>         Number of feature neurons"
        echo "  -c <class_neurons>           Number of class neurons"
        echo "  -o <output_weight>           Output weight"
        echo "  -b <num_bins>                Number of bins"
        echo "  -n <num_tests>               Number of reservoirs to generate (default: 10)"
        echo "  -w <output_file>             Output file for best reservoir (default: best_reservoir.json)"
        exit 1
        ;;
    esac
done

data_dir=${@:$OPTIND:1}

if [ -z "$data_dir" ]; then
    echo "Usage: $0 [options] <data_directory>"
    echo "Options:"
    echo "  -s <size>                    Reservoir size"
    echo "  -p <connection_probability>  Connection probability"
    echo "  -c <class_neurons>           Number of class neurons"
    echo "  -o <output_weight>           Output weight"
    echo "  -b <num_bins>                Number of bins"
    echo "  -n <num_tests>               Number of reservoirs to generate (default: 10)"
    exit 1
fi

data_range=$(bin/data_preprocessing <${data_dir}/data.csv)
num_features=$(grep 'Num' <<<${data_range} | awk '{print $2}')
label_count=$(sort -n <${data_dir}/labels.csv | uniq | wc -l)

best_seed=-1
deltas=()
best_min=0
for i in $(seq 1 $N); do
    seed=$((RANDOM % 65563))

    printf '\0331\rGenerating Reservoir %d/%d' $((i)) $((N))
    out=$(bash ./scripts/calculate_grade.bash -s "${s}" -p "${p}" -f $((num_features * num_bins)) -c "${c}" -o "${o}" -b "${num_bins}" -r ${seed} ${data_dir})

    if ! grep -q "INVALID" <<<"$out"; then
        if [[ $best_seed -eq -1 ]]; then
            best_seed=${seed}
            deltas=($(awk '{print $5}' <<<"$(tail -n $label_count <<<"$out")"))
            best_min=$(awk 'BEGIN {min = 1.0} {if ($5 < min) { min = $5 }} END {print min}' <<<"$(tail -n "${label_count}" <<<"${out}")")
        else
            tmp_deltas=($(awk '{print $5}' <<<"$(tail -n $label_count <<<"$out")"))
            tmp_min=$(awk 'BEGIN {min = 1.0} {if ($5 < min) { min = $5 }} END {print min}' <<<"$(tail -n "${label_count}" <<<"${out}")")
            if (($(bc -l <<<"$tmp_min > $best_min"))); then
                best_seed=${seed}
                deltas=(${tmp_deltas[@]})
            fi
        fi

    fi
done

printf '\n'

if [[ $best_seed -eq -1 ]]; then
    echo "No valid reservoir generated in $N tests."
    exit 1
fi

echo ""
echo "Best seed: $best_seed"
echo ""
echo "Smallest Deltas:"
for i in $(seq 0 $((label_count - 1))); do
    echo "Class $i: ${deltas[i]}"
done
echo ""

rm out.json
bin/generate_reservoir -s ${s} -p ${p} -f $((num_features * num_bins)) -c ${c} -o ${o} -r $best_seed | framework-open/bin/network_tool >${output_file}
