#!/bin/bash 

pdb="2KL8.pdb"
contigs='A1-20,5,A26-79'
out='out/2KL8_test'
num_designs=2
tmpl_conf=0.5
script='../inpaint.py'

mkdir -p `dirname $out`

python $script --pdb $pdb --contigs $contigs --dump_all --n_cycle 15 --out $out --num_designs $num_designs

