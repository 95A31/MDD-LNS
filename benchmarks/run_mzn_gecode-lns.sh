#!/usr/bin/bash

minizinc="./solvers/minizinc/bin/minizinc"
gecode="./solvers/gecode/bin/fzn-gecode"
gecode_cfg="./solvers/gecode/gecode.msc"

if [ "$#" -ne 2 ];
then
    echo "[ERROR]: Usage: bash $(basename $0) <Benchmark> <Timeout>."
    exit 1
fi

benchmark=$1
timeout_sec=$2
runs=10
model="./${benchmark}/${benchmark}_lns"
output_file="./output/${benchmark}_mzn_gecode-lns_${timeout_sec}.txt"

echo -n > ${output_file}

for data in $(ls ./${benchmark}/data/dzn/*.dzn); do
    filename=$(basename -- "$data")
    filename="${filename%.*}"
    for i in $(seq 1 $runs); do  
        echo -n -e "Solving ${filename} with Gecode LNS... (${i} / ${runs})\r"
        echo "%%%% Instance: ${filename}" >> ${output_file}
        echo "%%%% Run: ${i} / ${runs}" >> ${output_file}
        ${minizinc} --solver ${gecode_cfg} -c ${model}.mzn ${data}
        ${gecode} --c-d 1 --a-d 2 -time ${timeout_sec}000 -r ${i} -p $(nproc --all) -a  -s -restart constant -restart-scale 10000 ${model}.fzn | ${minizinc} --output-time --ozn-file ${model}.ozn >> ${output_file}
        rm -f ${model}.fzn ${model}.ozn
    done
    echo ""
done
