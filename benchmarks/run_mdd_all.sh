#!/usr/bin/bash

if [ "$#" -ne 1 ];
then
    echo "[ERROR]: Usage: bash $(basename $0) <Timeout>."
    exit 1
fi

timeout_sec=$1

for b in sop ctwp ossp; do
  bash run_mdd_cpu-only.sh ${b} ${timeout_sec};
  bash run_mdd_cpu-gpu.sh ${b} ${timeout_sec};
done;


    
