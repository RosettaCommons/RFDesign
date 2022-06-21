#!/usr/bin/env python
# 
# Extracts chain A from 2-chain hallucination output files (.pdb, .fas, .npz,
# .trb). Assumes that chain A is placed first.
#
# Moves the original 2-chain outputs to a subfolder and puts the new outputs
# where the original 2-chain files were.
#
# TODO: Implement case where chain A is second.
# 
# Usage:
#
#     ./extract_chainA_all_outputs.py FOLDER
#
# Updated 2022-1-6

import pandas as pd
import numpy as np
import sys, os, argparse, glob, shutil, pickle
from collections import OrderedDict

p = argparse.ArgumentParser()
p.add_argument('data_dir', help='Folder of TrDesign outputs to process')
p.add_argument('--out_dir', help='Folder to output chain A files. Default: FOLDER/chainA/')
args = p.parse_args()

if args.out_dir is None:
    args.out_dir = os.path.join(args.data_dir, 'chainA/')

os.makedirs(args.out_dir, exist_ok=True)

for fn in glob.glob(os.path.join(args.data_dir,'*.pdb')):
    name = os.path.basename(fn).replace('.pdb','')

    # remove chain B from pdb
    twochain_fn = fn
    monomer_fn = os.path.join(args.out_dir,name+'.pdb')
    if os.path.exists(monomer_fn):
        print(f'{monomer_fn} exists, skipping this design.')
        continue

    seen_ter = False # keep only 1 "TER" line
    with open(twochain_fn) as f:
        lines = f.readlines()
    with open(monomer_fn,'w') as outf:
        for line in lines:
            if len(line)>22 and line[21]=='B':
                pass
            elif seen_ter and 'TER' in line:
                pass
            else:
                if not seen_ter and 'TER' in line:
                    seen_ter = True
                print(line.strip(),file=outf)

    # remove chain B from fas
    twochain_fn = fn.replace('.pdb','.fas')
    monomer_fn = os.path.join(args.out_dir,name+'.fas')

    seq = open(twochain_fn).readlines()[1].strip()
    L = seq.index('/')
    seq = seq[:L]
    with open(monomer_fn,'w') as outf:
        print(f'>{name}', file=outf)
        print(seq, file=outf)

    # remove chain B from npz (only works for --rec_placement second)
    twochain_fn = fn.replace('.pdb','.npz')
    monomer_fn = os.path.join(args.out_dir,name+'.npz')

    npz = np.load(twochain_fn)
    npz_ = {x:npz[x][:L,:L] for x in npz}
    np.savez(monomer_fn, **npz_)

    # remove chain B from trb
    twochain_fn = fn.replace('.pdb','.trb')
    monomer_fn = os.path.join(args.out_dir,name+'.trb')

    trb = np.load(twochain_fn, allow_pickle=True)
    idx = [i for i,x in enumerate(trb['con_ref_pdb_idx']) if x[0]=='A']

    trb['con_ref_pdb_idx'] = [x for i,x in enumerate(trb['con_ref_pdb_idx']) if i in idx]
    trb['con_hal_pdb_idx'] = [x for i,x in enumerate(trb['con_hal_pdb_idx']) if i in idx]
    trb['con_ref_idx0'] = trb['con_ref_idx0'][idx]
    trb['con_hal_idx0'] = trb['con_hal_idx0'][idx]

    with open(monomer_fn,'wb') as outf:
        pickle.dump(trb, outf)
