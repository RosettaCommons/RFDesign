#!/bin/bash
mkdir /output

# Run hallucination
/root/miniconda3/envs/rfdesign-cuda/bin/python3 RFDesign/hallucination/hallucinate.py $@ --out /output/design

# Run rosetta relaxation
RFDesign/scripts/trfold_relax.sh /output/

# Run AF2 metrics
/root/miniconda3/envs/rfdesign-cuda/bin/python3 RFDesign/scripts/af2_metrics.py /output/trf_relax

# Run PyRosetta metrics
/root/miniconda3/envs/rfdesign-cuda/bin/python3 RFDesign/scripts/pyrosetta_metrics.py /output/trf_relax
