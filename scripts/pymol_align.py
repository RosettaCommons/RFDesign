#!/usr/bin/pymol -qc

#
# Aligns hallucination output models to template structures by contig
# (constrained motif) or interface residues
# 
# Usage:
#
#   ./pymol_align.py --template TEMPLATE_PDB PDB1 PDB2 ...
#
# Updated 2022-1-6

import pandas as pd
import numpy as np
from pymol import cmd
from glob import glob
from os.path import dirname, basename, join
import argparse

p = argparse.ArgumentParser()
p.add_argument('pdbs', nargs='+', help='PDB files to align to template')
p.add_argument('-t','--template', help='Template (natural binder) structure (.pdb)')
p.add_argument('-r','--receptor', help='Receptor structure (.pdb). Should be in same coordinate ' \
                                                                             'frame as template PDB.')
p.add_argument('-o','--out', required=True, help='Output filename (.pse)')
p.add_argument('--exclude_chain', type=str, help='Chain to exclude from alignment.')
p.add_argument('--trb_dir', help='Folder of of .trb files with indices of constrained regions.')
p.add_argument('--interface_res', help='Only align positions in this file (text file ' \
                                       'with space-delimited integers)')
p.add_argument('--pdb_suffix', default='', help='PDB files have this suffix relative to trb files')

p.add_argument('--show_sc', action='store_true', default=True, help='Turns on sidechains and formats them.')
p.add_argument('--show_hbonds', action='store_true', default=False, help='Shows intra-chain hydrogen bonds')
p.add_argument('--group_hbonds', action='store_true', default=False, help='Group structures and hbonds so they can be shown together in grid mode.')

p.add_argument('--nocolor', action='store_true', help='Do not color contigs in Pymol outputs.')
p.add_argument('--colors', nargs=4, default=['paleyellow','brightorange','gray80','purple'],
                             help='4 Pymol colors (space-delimited) for ref, ref contig, design, design contig.')
args = p.parse_args()

def parse_range(_range):
    if '-' in _range:
      s, e = _range.split('-')
    else:
      s, e = _range, _range

    return int(s), int(e)

def parse_contig(contig):
    '''
    Return the chain, start and end residue in a contig or gap str.

    Ex:
    'A4-8' --> 'A', 4, 8
    'A5'   --> 'A', 5, 5
    '4-8'  --> None, 4, 8
    'A'    --> 'A', None, None
    '''

    # is contig
    if contig[0].isalpha():
      ch = contig[0]
      if len(contig) > 1:
        s, e = parse_range(contig[1:])
      else:
        s, e = None, None
    # is gap
    else:
      ch = None
      s, e = parse_range(contig)

    return ch, s, e

def expand(mask_str):
    '''
    Ex: '2,A3-5,3' --> [None, None, (A,3), (A,4), (A,5), None, None, None]
    '''
    expanded = []
    for l in mask_str.split(','):
      ch, s, e = parse_contig(l)

      # contig
      if ch:
        expanded += [(ch, res) for res in range(s, e+1)]
      # gap
      else:
        expanded += [None for _ in range(s)]

    return expanded

def idx2contigstr(idx):
    istart = 0
    contigs = []
    for iend in np.where(np.diff(idx)!=1)[0]:
        contigs += [f'{idx[istart]}-{idx[iend]}']
        istart = iend+1
    contigs += [f'{idx[istart]}-{idx[-1]}']
    return contigs

### Main  ###
if args.template is not None:
    cmd.load(args.template)
    ref_prefix = f'/{os.path.basename(args.template).replace(".pdb","")}//'
else:
    # load reference pdb from first trb
    pdbfile = args.pdbs[0]
    trbfile = glob(join(os.path.dirname(pdbfile), 
                        basename(pdbfile).replace(args.pdb_suffix+'.pdb','.trb')))[0]
    trb = np.load(trbfile,allow_pickle=True)
    refpdb = trb['settings']['pdb']
    cmd.load(refpdb)
    ref_prefix = f'/{os.path.basename(refpdb).replace(".pdb","")}//'

if args.receptor is not None: 
    cmd.load(args.receptor)

for pdbfile in args.pdbs:
    name = basename(pdbfile.replace('.pdb',''))
    if args.trb_dir is None: trb_dir = os.path.dirname(pdbfile)
    else: trb_dir = args.trb_dir
    print(join(trb_dir,name+'.trb'))

    trbfile = glob(join(trb_dir, basename(pdbfile).replace(args.pdb_suffix+'.pdb','.trb')))[0]
    trb = np.load(trbfile,allow_pickle=True)

    cmd.load(pdbfile)

    pdb_idx_ref = trb['con_ref_pdb_idx']
    pdb_idx_hal = trb['con_hal_pdb_idx']

    if args.exclude_chain is not None:
        print(f'Excluding chain {args.exclude_chain} from aligned region.')
        pdb_idx_hal = [y for x,y in zip(pdb_idx_ref,pdb_idx_hal) if x[0] != args.exclude_chain]
        pdb_idx_ref = [x for x in pdb_idx_ref if x[0] != args.exclude_chain]

    if args.interface_res is not None:
        pdb_idx_interface = expand(args.interface_res)
        i_keep = [i for i,idx in enumerate(trb['con_ref_pdb_idx']) if idx in pdb_idx_interface]
        pdb_idx_ref = [idx for i,idx in enumerate(trb['con_ref_pdb_idx']) if i in i_keep]
        pdb_idx_hal = [idx for i,idx in enumerate(trb['con_hal_pdb_idx']) if i in i_keep]

    ref_coords = [f'{ref_prefix}{ch}/{i}/CA' for ch,i in pdb_idx_ref]
    hal_coords = [f'{name}//{ch}/{i}/CA' for ch,i in pdb_idx_hal]
    pairs = [x for pair in zip(hal_coords,ref_coords) for x in pair]
    rmsd = cmd.pair_fit(*pairs)
 
    sele = ' or '.join([x.replace('/CA','/') for x in hal_coords])
    cmd.select(f'contigs_{name}_{rmsd:.2f}', sele)
    if not args.nocolor:
        cmd.color(args.colors[2],f'{name}')
        cmd.color(args.colors[3],f'contigs_{name}_{rmsd:.2f}')

    if args.show_hbonds:
        cmd.dist(name+'_hbonds',f"{name} and not (solvent)",f"{name} and not (solvent)",quiet=1,mode=2,label=0,reset=1)
        cmd.enable(name+'_hbonds')
        if args.group_hbonds:
            cmd.group(name+'_group',name)
            cmd.group(name+'_group',name+'_hbonds',quiet=0)

cmd.select(f'contigs_ref', ' or '.join([f'{ref_prefix}{ch}/{i}' for ch,i in pdb_idx_ref]))
if not args.nocolor:
    cmd.color(args.colors[0],ref_prefix)
    cmd.color(args.colors[1],f'contigs_ref')
if args.show_hbonds:
    refname = ref_prefix.replace('/','')
    cmd.dist(refname+'_hbonds',f"{refname} and not (solvent)",f"{refname} and not (solvent)",quiet=1,mode=2,label=0,reset=1)
    cmd.enable(refname+'_hbonds')
    if args.group_hbonds:
        cmd.group(refname+'_group',refname)
        cmd.group(refname+'_group',refname+'_hbonds',quiet=0)
cmd.zoom(ref_prefix)
cmd.deselect()

if args.show_sc:
    cmd.show("sticks")
    cmd.hide("(bb.&!(n. CA|n. N&r. PRO))")
    cmd.hide("(hydro and (elem C extend 1))")
    util.cnc("all",_self=cmd)
#cmd.set('surface_cavity_mode',1,'',0)
#cmd.show('surface')

cmd.save(args.out)
