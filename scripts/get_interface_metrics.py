#!/software/conda/envs/pyrosetta/bin/python3.7
#
# Calculates contact molecular surface and various other metrics suggested by
# Brian Coventry. XML is unaltered from Brian.
#
# Usage:
#
#   ./get_interface_metrics.py FOLDER
#
# where FOLDER contains PDBs of complexes you want to score.
#

import pandas as pd
import numpy as np
import os, glob, argparse
from collections import OrderedDict
import multiprocessing

import pyrosetta
from pyrosetta import *
from rosetta.protocols.rosetta_scripts import *

script_dir = os.path.dirname(__file__)

p = argparse.ArgumentParser()
p.add_argument('data_dir', help='Folder of FastDesign outputs to process. Must contain both binder and receptor.')
p.add_argument('--suffix', default='', help='Suffix on outputs to process. e.g. if --suffix=complex, then only processes files named *complex.pdb')
p.add_argument('--out', help='Output file name. Default: interface_metrics.csv')
args = p.parse_args()
if args.out is None: args.out = os.path.join(args.data_dir, 'interface_metrics.csv')

init('-corrections::beta_nov16 -holes:dalphaball /software/rosetta/DAlphaBall.gcc -detect_disulf false -run:preserve_header true')
parser = RosettaScriptsParser()
protocol = parser.generate_mover(script_dir+"/interface_metrics.xml")

ncpu = len(os.sched_getaffinity(0))
print(f'Using {ncpu} cores')

def calculate(fn):
    pose = pose_from_pdb(fn)
    protocol.apply(pose)
    row = OrderedDict([(x,pose.scores[x]) for x in ['binder_blocked_sap',
     'buns_heavy_ball_1.1D',
     'contact_molec_sq5_apap_target',
     'contact_molecular_surface',
     'contact_molecular_surface_ap_target',
     'contact_molecular_surface_apap_target',
     'ddg',
     'interface_buried_sasa',
     'interface_sc',
     'mismatch_probability',
     'sap_score',
     'sap_score_target',
     'target_blocked_sap']])
    row['name'] = os.path.basename(fn).replace('.pdb','')
    return row

files = glob.glob(os.path.join(args.data_dir, f'*{args.suffix}.pdb'))
with multiprocessing.Pool(ncpu) as p:
    records = p.map(calculate, files)
df = pd.DataFrame.from_records(records)
df.to_csv(args.out)

