#!/bin/bash
if [ -z "$2" ]; then
    jobname=$1
else
    jobname=$2
fi

sbatch -a 1-$(cat $1 | wc -l) -p gpu -J $jobname \
       -c 2 --mem=12g --gres=gpu:rtx2080:1 \
       --wrap="eval \`sed -n \${SLURM_ARRAY_TASK_ID}p $1\`" \
