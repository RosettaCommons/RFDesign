#!/bin/bash
mkdir /output
/root/miniconda3/envs/rfdesign-cuda/bin/python3 RFDesign/hallucination/hallucinate.py $@ --out /output/design
