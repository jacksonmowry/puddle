#!/usr/bin/env bash

set -euo pipefail

s="250"
p="0.025"
c="64"
o="0.3"
N=10
num_bins="10"
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

data_dirs=(${@:$OPTIND})
echo "${data_dirs[@]}"

if [ -z "${data_dirs[*]}" ]; then
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

max_num_features=$( (for file in "${data_dirs[@]}"; do bin/data_preprocessing <"${file}"/data.csv; done) | grep Num | sort -k2 -n | tail -1 | awk '{print $2}')
declare -a label_counts=()

for file in "${data_dirs[@]}"; do
    label_counts+=($(sort -n <"${file}/labels.csv" | uniq | wc -l))
done

best_seed=-1
best_composite_score=9999
intraclass_factor="0.4"
intraclass="0"
interclass="0"
for i in $(seq 1 $N); do
    seed=$((RANDOM % 65563))

    printf '\0330\rGenerating Reservoir %d/%d, Best so far = %s (want to minimize this term)' $((i)) $((N)) "${best_composite_score}"

    reservoir_worst_score="-9999.0"
    i=0

    intra_tmp="0"
    inter_tmp="0"

    for file in "${data_dirs[@]}"; do
        out=$(bash ./scripts/calculate_grade.bash -s "${s}" -p "${p}" -f $((max_num_features * num_bins)) -c "${c}" -o "${o}" -b "${num_bins}" -r ${seed} "${file}")

        deltas=($(awk '{print $5}' <<<"$(tail -n "${label_counts[i]}" <<<"${out}")"))
        inter=$(awk 'BEGIN {min = 999} {if ($5 < min) { min = $5 }} END { print min }' <<<"$(tail -n "${label_counts[i]}" <<<"${out}")")
        intra=$(awk '{if ($1 == "Maximum") { print $4 }}' <<<"${out}")
        composite_score=$(bc -l <<<"-1 * ${inter} * (1 - ${intraclass_factor}) + ${intra} * ${intraclass_factor}")

        if (($(bc -l <<<"${composite_score} > ${reservoir_worst_score}"))); then
            reservoir_worst_score="${composite_score}"
            intra_tmp="${intra}"
            inter_tmp="${inter}"
        fi

        i=$((i + 1))
    done

    if (($(bc -l <<<"${reservoir_worst_score} < "${best_composite_score}))); then
        best_composite_score="${reservoir_worst_score}"
        best_seed="${seed}"
        intraclass="${intra_tmp}"
        interclass="${inter_tmp}"
    fi
done

printf '\n'

if [[ $best_seed -eq -1 ]]; then
    echo "No valid reservoir generated in $N tests."
    exit 1
fi

echo ""
echo "Best seed: ${best_seed} Composite score: ${best_composite_score} Intra: ${intraclass} Inter: ${interclass}"

bin/generate_reservoir -s ${s} -p ${p} -f $((max_num_features * num_bins)) -c ${c} -o ${o} -r $best_seed | framework-open/bin/network_tool >${output_file}
