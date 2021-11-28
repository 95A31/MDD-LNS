#!/usr/bin/bash

if [ "$#" -ne 2 ];
then
    echo "[ERROR]: Usage: bash $(basename $0) <Benchmark> <Timeout>."
    exit 1
fi

benchmark=$1
timeout_sec=$2
output_file="./output/${benchmark}_mdd_cpu-gpu_${timeout_sec}.txt"
runs=3

# Compile mdd-gpu
sed -i "24s/.*/using ProblemType = ${benchmark^^}roblem;/" ../src/Main.cu
sed -i "25s/.*/using StateType = ${benchmark^^}State;/" ../src/Main.cu
cmake --build ../cmake-build-remote-release/

echo -n > ${output_file}
for data in ./${benchmark}/data/json/*.json; do
    filename=$(basename -- "$data")
    filename="${filename%.*}"
    for i in $(seq 1 $runs); do  
        echo -n -e "Solving ${filename} with MDD (CPU + GPU) ... (${i} / ${runs})\r"
        echo "%%%% Instance: ${filename}" >> ${output_file}
        echo "%%%% Run: ${i} / ${runs}" >> ${output_file}
        ../cmake-build-remote-release/mdd-gpu -t ${timeout_sec} --wc 1000 --mc 16 --wg 5 --mg 10000 --eq 0.7 --neq 0 --rs ${i} ${data} >> ${output_file}
    done
    echo ""
done
