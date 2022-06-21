#!/bin/bash
#
# Unit tests for hallucination design pipeline
#

# exit when any command fails
set -e
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

# absolute path of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" 

#########################
# Command-line options
#########################
OPTIND=1
to_run=all
while getopts ":1234" opt; do
    case $opt in
        1) to_run=1;;
        2) to_run=2;;
        3) to_run=3;;
        4) to_run=4;;
        all) to_run=all;;
    esac
done
shift $((OPTIND-1))


rm -rf output
mkdir -p output

#########################
# C3d
#########################

i=1
if [[ "$to_run" == "all" || "$to_run" == "1" ]]; then
    echo "Running test $i (C3d)..."
    python $DIR/../hallucinate.py --out output/test${i} \
        --pdb $DIR/input/C3d_relaxed.pdb --len 80-90 --contigs A104-126,A170-185 \
        --steps g20,m3 --exclude_aa C \
        --force_aa A112K,A116L,A117E,A119Q,A120K,A121P,A122D,A123G,A171S,A174G,A175S,A177T,A178K,A181D,A185A \
        --w_cce 1 --w_entropy 0 --w_kl 1 --use_template True --num 2 --drop 0  

    count=`ls $DIR/output/test${i}_*.npz | wc -w`
    if [ ! "$count" -eq 2 ]; then
        echo "ERROR: Test $i failed to generate the expected number of outputs"
        exit 1
    fi
    echo "Test $i succeeded!"
    echo
fi

#########################
# PD1
#########################
i=$((i+1))
if [[ "$to_run" == "all" || "$to_run" == "$i" ]]; then
    echo "Running test $i (PD1)..."
    python $DIR/../hallucinate.py --pdb=$DIR/input/pd1.pdb --out=output/test$i \
                      --num=2 --start_num=0 \
                      --mask=15,A119-140,15,A63-82,5 --steps=g20 \
                      --w_crmsd 0.5 --save_pdb=True --track_step=10

    if [ ! -f $DIR/output/test${i}_1.npz ]; then
        echo "ERROR: Test $i failed to generate the expected number of outputs"
        exit 1
    fi
    echo "Test $i succeeded!"
    echo
fi

#########################
# rep, atr, rog losses
#########################
i=$((i+1))
if [[ "$to_run" == "all" || "$to_run" == "$i" ]]; then
    echo "Running test $i (repulsive, attractive, and Rg losses)..."
    python $DIR/../hallucinate.py --pdb=$DIR/input/rsvf-v_5tpn.pdb \
                      --network_name rf_Nov05_2021 \
                      --out=output/test$i --num=2 \
                      --contigs=A163-181 --len=60 --steps=g20 \
                      --w_rep=1 --rep_sigma=4 --rep_pdb=$DIR/input/rsvf-v_5tpn_receptor_frag.pdb \
                      --w_atr=10 --atr_sigma=6 --atr_pdb=$DIR/input/rsvf-v_5tpn_receptor_frag.pdb \
                      --save_pdb=True 

    if [ ! -f $DIR/output/test${i}_1.npz ]; then
        echo "ERROR: Test $i failed to generate the expected number of outputs"
        exit 1
    fi
    echo "Test $i succeeded!"
    echo
fi

echo "All tests completed successfully!"
