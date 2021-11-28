#!/usr/bin/bash

minizinc="./solvers/minizinc/bin/minizinc"
yuck="./solvers/yuck/bin/yuck"
yuck_cfg="./solvers/yuck/yuck.msc"

if [ "$#" -ne 2 ];
then
    echo "[ERROR]: Usage: bash $(basename $0) <Benchmark> <Timeout>."
    exit 1
fi

benchmark=$1
timeout_sec=$2
runs=10
model="./${benchmark}/${benchmark}_original"
output_file="./output/${benchmark}_mzn_yuck-original_${timeout_sec}.txt"

echo -n > ${output_file}

for data in $(ls ./${benchmark}/data/dzn/*.dzn); do
    filename=$(basename -- "$data")
    filename="${filename%.*}"
    for i in $(seq 1 $runs); do  
        echo -n -e "Solving ${filename} with Yuck... (${i} / ${runs})\r"
        echo "%%%% Instance: ${filename}" >> ${output_file}
        echo "%%%% Run: ${i} / ${runs}" >> ${output_file}
        ${minizinc} --solver ${yuck_cfg} -c ${model}.mzn ${data}
        ${yuck} --runtime-limit ${timeout_sec} -r ${i} -p $(nproc --all) -a --restart-limit 10000 --run-presolver false ${model}.fzn | ${minizinc} --output-time --ozn-file ${model}.ozn >> ${output_file}
        rm -f ${model}.fzn ${model}.ozn
    done
    echo ""
done