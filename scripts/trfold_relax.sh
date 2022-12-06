#!/bin/bash
#
# Performs relax on TRFold structures.
#
# Usage:
#
#   ./trfold_relax.sh FOLDER            # puts on short queue
#
#   ./trfold_relax.sh FOLDER backfill   # puts on backfill queue
#
# will relax all pdb files in folders, using restraints derived from matching
# npz files, and output relaxed structures into a subfolder called trf_relax/.
#
​
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
​
OPTIND=1
roll="--roll"
while getopts ":n" opt; do
    case $opt in
        n) roll="";;
    esac
done
shift $((OPTIND-1))
​
if [ -z "$2" ]; then
    queue=short
else
    queue=$2
fi
​
outdir=$1/trf_relax
mkdir -p $outdir
task_file=`basename $1`.fold.list
for PDB in $1/*.pdb; do
  f=`basename $PDB .pdb`
  NPZ=`dirname $PDB`/$f.npz
  if [ ! -f $outdir/$f.pdb ]; then
    echo "Relaxing $PDB"
    /root/miniconda3/envs/rfdesign-cuda/bin/python3 $DIR/RosettaTR/trfold_relax.py $roll -sg 7,3 $NPZ $PDB $outdir/$f.pdb
  fi
done
​
cd $outdir
ct=$(find ../ -maxdepth 1 -name '*.trb' | wc -l)
if [ "$ct" -gt 0 ]; then
    ln -sf ../*.trb .
fi