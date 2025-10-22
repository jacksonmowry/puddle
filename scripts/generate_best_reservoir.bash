#!/usr/bin/env bash

data_dir=''
s="-s 250"
p="-p 0.025"
f=-1
c="-c 64"
o="-o 0.3"
N=10
num_bins="-b 10"
output_file="best_reservoir.json"

while getopts "ds:p:f:c:o:r:b:n:w:" opt; do
    case ${opt} in
        s )
            s="-s ${OPTARG}"
            ;;
        p )
            p="-p ${OPTARG}"
            ;;
        f )
            f="-f ${OPTARG}"
            ;;
        c )
            c="-c ${OPTARG}"
            ;;
        o )
            o="-o ${OPTARG}"
            ;;
        b )
            num_bins="-b ${OPTARG}"
            ;;
        n )
            N=${OPTARG}
            ;;
        w )
            output_file="${OPTARG}"
            ;;
        \? )
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
    echo "  -f <feature_neurons>         Number of feature neurons"
    echo "  -c <class_neurons>           Number of class neurons"
    echo "  -o <output_weight>           Output weight"
    echo "  -b <num_bins>                Number of bins"
    echo "  -n <num_tests>               Number of reservoirs to generate (default: 10)"
    exit 1
fi

data_range=$(bin/data_preprocessing <${data_dir}/data.csv)
num_features=$(grep 'Num' <<<${data_range} | awk '{print $2}')
label_count=$(sort -n <${data_dir}/labels.csv | uniq | wc -l)

if [ ${f} -eq -1 ]; then
    f=$((10 * num_features))
    f="-f ${f}"
fi

best_seed=-1
deltas=()
for i in $(seq 1 $N); do
    seed=$(od -An -N4 -tu4 < /dev/urandom)

    echo "Generating Reservoir $i/$N"
    out=$(bash ./scripts/calculate_grade.bash ${s} ${p} ${f} ${c} ${o} ${num_bins} -r ${seed} ${data_dir})

    if ! grep -q "INVALID" <<<"$out"; then
        if [[ $best_seed -eq -1 ]]; then
            best_seed=${seed}
            deltas=($(awk '{print $5}' <<<"$(tail -n $label_count <<<"$out")"))
        else
            tmp_deltas=($(awk '{print $5}' <<<"$(tail -n $label_count <<<"$out")"))
            best_avg=0
            cur_avg=0
            for j in $(seq 0 $((label_count - 1))); do
                best_avg=$(bc -l <<< "${best_avg} + ${deltas[j]} / ${label_count}")
                cur_avg=$(bc -l <<< "${cur_avg} + ${tmp_deltas[j]} / ${label_count}")
            done
            if (( $(bc -l <<< "$cur_avg > $best_avg") )); then
                best_seed=${seed}
                deltas=(${tmp_deltas[@]})
            fi
        fi

    fi
done

if [[ $best_seed -eq -1 ]]; then
    echo "No valid reservoir generated in $N tests."
    exit 1
fi

echo "Best seed: $best_seed"
echo "Smallest Deltas:"
for i in $(seq 0 $((label_count - 1))); do
    echo "Class $i: ${deltas[i]}"
done

rm out.json
bin/generate_reservoir ${s} ${p} ${f} ${c} ${o} -r $best_seed | framework-open/bin/network_tool > ${output_file}