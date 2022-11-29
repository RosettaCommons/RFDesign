#!/bin/bash
conda activate rfdesign-cuda
pwd
python3 hallucination/hallucinate.py $@
