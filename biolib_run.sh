#!/bin/bash
conda activate rfdesign-cuda
python3 hallucination/hallucinate.py $@
