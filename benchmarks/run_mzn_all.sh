#!/usr/bin/bash

if [ "$#" -ne 1 ];
then
    echo "[ERROR]: Usage: bash $(basename $0) <Timeout>."
    exit 1
fi

timeout_sec=$1

for b in sop ctwp ossp; do
    bash run_mzn_gecode-original.sh ${b} ${timeout_sec};
    bash run_mzn_gecode-lns.sh ${b} ${timeout_sec};
    bash run_mzn_yuck-original.sh ${b} ${timeout_sec};
done;
    
