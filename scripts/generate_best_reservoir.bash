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
best_max=0
composite_score=0
intraclass_factor="0.0"
for i in $(seq 1 $N); do
    seed=$((RANDOM % 65563))

    printf '\0330\rGenerating Reservoir %d/%d' $((i)) $((N))
    if ((i != 1)); then
        printf ' %s' "${deltas[*]}"
        printf ' %s' "${best_max}"
        printf ' (%s)' "${composite_score}"
    fi

    out=$(bash ./scripts/calculate_grade.bash -s "${s}" -p "${p}" -f $((num_features * num_bins)) -c "${c}" -o "${o}" -b "${num_bins}" -r ${seed} ${data_dir})

    if ! grep -q "INVALID" <<<"$out"; then
        if [[ $best_seed -eq -1 ]]; then
            best_seed=${seed}
            deltas=($(awk '{print $5}' <<<"$(tail -n $label_count <<<"$out")"))
            best_min=$(awk 'BEGIN {min = 9999} {if ($5 < min) { min = $5 }} END {print min}' <<<"$(tail -n "${label_count}" <<<"${out}")")
            best_max=$(awk '{if ($1 == "Maximum") {print $4}}' <<<"${out}")

            # We attempting to maximize the interclass angle and minimize the intraclass angle
            # We negate the interclass angle so that we can minimize the entire problem
            composite_score=$(bc -l <<<"-1 * ${best_min} * (1 - ${intraclass_factor}) + ${best_max} * ${intraclass_factor}")
        else
            tmp_deltas=($(awk '{print $5}' <<<"$(tail -n $label_count <<<"$out")"))
            tmp_min=$(awk 'BEGIN {min = 9999} {if ($5 < min) { min = $5 }} END {print min}' <<<"$(tail -n "${label_count}" <<<"${out}")")
            tmp_max=$(awk '{if ($1 == "Maximum") {print $4}}' <<<"${out}")
            tmp_composite=$(bc -l <<<"-1 * ${tmp_min} * (1 - ${intraclass_factor}) + ${tmp_max} * ${intraclass_factor}")
            printf ' | Found %s (Intra: %s, Inter: %s)\n' "${tmp_composite}" "${tmp_max}" "${tmp_min}"
            if (($(bc -l <<<"$tmp_min != 9999 && $tmp_composite < $composite_score"))); then
                best_seed=${seed}
                best_min="${tmp_min}"
                best_max="${tmp_max}"
                deltas=(${tmp_deltas[@]})
                composite_score="${tmp_composite}"
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

bin/generate_reservoir -s ${s} -p ${p} -f $((num_features * num_bins)) -c ${c} -o ${o} -r $best_seed | framework-open/bin/network_tool >${output_file}
