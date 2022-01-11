#!/usr/bin/env python
# 
# Extracts chain A from a pdb file.
# 
# Usage:
#
#     ./extract_chainA_pdb_only.py FOLDER
#
# This will output chain-A-only pdbs to FOLDER/chainA/
#
# Updated 2022-1-8

import sys, os, argparse, glob, shutil

p = argparse.ArgumentParser()
p.add_argument('data_dir', help='Folder of TrDesign outputs to process')
p.add_argument('-o','--out_dir', help='Output folder.')
p.add_argument('--out_suffix', default='', help='Suffix for output files')
args = p.parse_args()

files = glob.glob(os.path.join(args.data_dir,'*pdb'))

if args.out_dir is None:
    args.out_dir = os.path.join(args.data_dir, 'chainA')

os.makedirs(args.out_dir, exist_ok=True)

for fn in files:
    print(fn)
    name = os.path.basename(fn).replace('.pdb','')

    # remove chain B from pdb
    twochain_fn = fn
    monomer_fn = os.path.join(args.out_dir,name+'.pdb')

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

