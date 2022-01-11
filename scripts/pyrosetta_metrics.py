#!/software/conda/envs/pyrosetta/bin/python
# 
# Calculates motif RMSDs between design models and reference structure for a
# folder of PDBs and outputs to a CSV. Also calculates radius of gyration,
# length, secondary structure fraction, and adds losses from trb file to CSV. 
#
# Usage:
#
#     ./pyrosetta_metrics.py DESIGNS_FOLDER
#
# This calculates the RMSD to the template given in the .trb file for each
# design. If you'd like to provide the template, use:
#
#     ./pyrosetta_metrics.py --template TEMPLATE_PDB DESIGNS_FOLDER
#
# Passing a file with space-delimited residue numbers to --interface_res will
# also output an 'interface_rmsd' on only the residues in the file (and in the
# contigs in the .trb file):
#
#     ./pyrosetta_metrics.py --template TEMPLATE_PDB --interface_res RES_FILE
#     DESIGNS_FOLDER
#
# Updated 2021-7-8

import pandas as pd
import numpy as np
import sys, os, argparse, glob
from collections import OrderedDict

import pyrosetta
pyrosetta.init('-mute all')

p = argparse.ArgumentParser()
p.add_argument('data_dir', help='Folder of TrDesign outputs to process')
p.add_argument('-t','--template', help='Template (natural binder) structure (.pdb)')
p.add_argument('--sc_rmsd', action='store_true', default=True, 
    help='Also calculate side-chain RMSD, returning NaN if residues aren\'t matched.')
p.add_argument('--interface_res',
    help='Comma-separated chain + residue ranges in reference pdb (e.g. "A13,A55-62"). '\
         'Report rmsd on these residues as "interface_rmsd"')
p.add_argument('-o','--out', help=('Prefix of output filenames.'))
p.add_argument('--trb_dir', help='Folder containing .trb files (if not same as pdb folder)')
p.add_argument('--pdb_suffix', default='', help='PDB files have this suffix relative to trb files')
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

def calc_rmsd(xyz_ref, xyz_hal, eps=1e-6):

    # center to CA centroid
    xyz_ref = xyz_ref - xyz_ref.mean(0)
    xyz_hal = xyz_hal - xyz_hal.mean(0)

    # Computation of the covariance matrix
    C = xyz_hal.T @ xyz_ref

    # Compute optimal rotation matrix using SVD
    V, S, W = np.linalg.svd(C)

    # get sign to ensure right-handedness
    d = np.ones([3,3])
    d[:,-1] = np.sign(np.linalg.det(V)*np.linalg.det(W))

    # Rotation matrix U
    U = (d*V) @ W

    # Rotate xyz_hal
    rP = xyz_hal @ U

    L = rP.shape[0]
    rmsd = np.sqrt(np.sum((rP-xyz_ref)*(rP-xyz_ref), axis=(0,1)) / L + eps)

    return rmsd

def mk_con_to_set(mask, set_id=None):
    '''
    Maps a mask or list of contigs to a set_id

    Input
    -----------
    mask (str): Mask or list of contigs. Ex: 3,B6-11,12,A12-19,9 or Ex: B6-11,A12-19
    set_id (list): List of integers Ex: [0,1]

    Output
    -----------
    con_to_set (dict): Maps contig str to integer
    '''

    # Extract contigs
    cons = [l for l in mask.split(',') if l[0].isalpha()]

    # Assign all contigs to one set if set_id is not passed
    if set_id is None:
        set_id = [0] * len(cons)

    con_to_set = dict(zip(cons, set_id))

    return con_to_set  
  
def get_rmsd(pose_ref, pose_hal, trb, mode='bb', interface_res=None):
    '''
    Inputs:
    ---------------
    mode (str; <bb, sc, fa>): Calculate rmsd over backbone, side chain or all heavy atoms (bb and sc).
    
    '''
    is_sc = pyrosetta.rosetta.core.scoring.is_protein_sidechain_heavyatom

    i_motif = [i for i,idx in enumerate(trb['con_ref_pdb_idx']) if idx[0]!='R']
    pdb_idx_ref = [idx for i,idx in enumerate(trb['con_ref_pdb_idx']) if i in i_motif]
    pdb_idx_hal = [idx for i,idx in enumerate(trb['con_hal_pdb_idx']) if i in i_motif]
    ref_to_hal = dict(zip(pdb_idx_ref, pdb_idx_hal))
    
    using_consets = 'con_set_id' in trb['settings'] and trb['settings']['con_set_id'] is not None
    if using_consets:
        # grab contigs and their sets
        if trb['settings'].get('mask') is not None:
            mask = trb['settings'].get('mask')
        elif trb['settings'].get('contigs') is not None:
            mask = trb['settings'].get('contigs')
            
        con_set_id = trb['settings'].get('con_set_id')
        if con_set_id is not None:
            con_set_id = con_set_id.split(',')
            
        con_to_set = mk_con_to_set(mask=mask, set_id=con_set_id)
    else:
        # if outputs are from autofold in window mode
        con_to_set = {'':0}
            
    # calc RMSD for every contig set
    squares_by_set = []
    lengths_by_set = []
    for set_id in set(con_to_set.values()):
        if using_consets:
            # Grab all ref_pdb_idx in the set_id
            set_ref_pdb_idx = []
            for k,v in con_to_set.items():
                if v == set_id:
                    ref_ch = k[0]
                    s, e = k[1:].split('-')
                    s, e = int(s), int(e)

                    for ref_idx in range(s, e+1):
                        set_ref_pdb_idx.append((ref_ch, ref_idx))
        else:
            set_ref_pdb_idx = pdb_idx_ref
            
        # Make alignment maps
        align_map = pyrosetta.rosetta.std.map_core_id_AtomID_core_id_AtomID()    
        for idx_ref in set_ref_pdb_idx:
            idx_hal = ref_to_hal[idx_ref]
          
            if interface_res is not None and idx_ref not in interface_res:
                continue

            # Find equivalent residues in both structures
            pose_idx_ref = pose_ref.pdb_info().pdb2pose(*idx_ref)
            pose_idx_hal = pose_hal.pdb_info().pdb2pose(*idx_hal)

            res_hal = pose_hal.residue(pose_idx_hal)
            res_ref = pose_ref.residue(pose_idx_ref)
            
            # Decide what atoms to calc rmsd over
            # sc
            if mode == 'sc':
                if res_hal.name3() != res_ref.name3():
                    print(f'Returning sidechain RMSD=NaN because template {res_ref.name1()}{idx_ref[1]} '\
                          f'!= hallucination {res_hal.name1()}{idx_hal[1]}.')
                    return np.nan
                
                atoms_sc = [res_ref.atom_name(i) for i in range(4,res_ref.natoms()+1) if is_sc(pose_ref, pose_ref, pose_idx_ref, i)]

            # bb
            atoms_bb = ['N', 'CA', 'C']
            
            if mode == 'fa':
              atoms = atoms_bb + atoms_sc
            elif mode == 'bb':
              atoms = atoms_bb
            elif mode == 'sc':
              atoms = atoms_sc
                
            # fill out alignment map
            for atom in atoms:
                atom_index = res_hal.atom_index(atom)  # this is the same number for either residue
                atom_id_ref = pyrosetta.rosetta.core.id.AtomID(atom_index, pose_idx_ref)
                atom_id_hal = pyrosetta.rosetta.core.id.AtomID(atom_index, pose_idx_hal)
                align_map[atom_id_hal] = atom_id_ref
        
        # Align and update metrics
        rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(pose_hal, pose_ref, align_map)
        L = len(atoms)
        lengths_by_set.append(L)
        squares_by_set.append((rmsd ** 2) * L)
    
    # Average RMSD across all sets
    lengths_by_set = np.array(lengths_by_set)
    squares_by_set = np.array(squares_by_set)
    rmsd_all_sets = np.sqrt( squares_by_set.sum() / lengths_by_set.sum() )
    
    return rmsd_all_sets

def get_topology(ss):
    topology = ''
    prev = None
    for x in ss:
        if (x != prev) and (x != 'L'):
            topology += x
        prev = x
    return topology

def main():

    # input and output names
    if args.out is not None:
        outfile = args.out if '.csv' in args.out else args.out+'.csv'
    else:
        outfile = os.path.join(args.data_dir,'pyrosetta_metrics.csv')

    # for radius of gyration
    rog_scorefxn = pyrosetta.ScoreFunction()
    rog_scorefxn.set_weight( pyrosetta.rosetta.core.scoring.ScoreType.rg , 1 )

    DSSP = pyrosetta.rosetta.protocols.moves.DsspMover()

    if args.template is not None:
        pose_ref_clean = pyrosetta.pose_from_file(args.template)
    else:
        last_template = ''

    # calculate contig RMSD
    print(f'Calculating RMSDs')

    df = pd.DataFrame()
    if args.trb_dir is not None: trb_dir = args.trb_dir
    else: trb_dir = args.data_dir

    records = []
    for fn in sorted(glob.glob(os.path.join(args.data_dir,'*.pdb'))):
        row = OrderedDict()
        row['name'] = os.path.basename(fn).replace('.pdb','')
        print(row['name'])

        trbname = os.path.join(trb_dir, os.path.basename(fn.replace(args.pdb_suffix+'.pdb','.trb')))
        if not os.path.exists(trbname): 
            sys.exit(f'ERROR: {trbname} does not exist. Set the --trb_dir argument if your .trb files '\
                      'are in a different folder from the .pdb files.')
        trb = np.load(trbname,allow_pickle=True)

        # save final losses
        for k in trb:
            if k.startswith('loss'):
                if type(trb[k]) is list:
                    row[k] = trb[k][0]
                else:
                    row[k] = trb[k]

        if args.template is None and 'settings' in trb and trb['settings']['pdb'] != last_template:
            pose_ref_clean = pyrosetta.pose_from_file(trb['settings']['pdb'])
            last_template = trb['settings']['pdb']

        pose_ref = pose_ref_clean.clone()
        pose_hal = pyrosetta.pose_from_file(fn)

        row['contig_rmsd'] = get_rmsd(pose_ref, pose_hal, trb, mode='bb')
        print('contig_rmsd: ', row['contig_rmsd'])
        if args.sc_rmsd:
            row['contig_sc_rmsd'] = get_rmsd(pose_ref, pose_hal, trb, mode='sc')
            print('contig_sc_rmsd: ', row['contig_sc_rmsd'])

        if args.interface_res is not None:
            row['interface_rmsd'] = get_rmsd(pose_ref, pose_hal, trb, mode='bb', 
                                             interface_res=expand(args.interface_res))
            print('interface_rmsd: ', row['interface_rmsd'])
            if args.sc_rmsd:
                row['interface_sc_rmsd'] = get_rmsd(pose_ref, pose_hal, trb, mode='sc',
                                                    interface_res=expand(args.interface_res))
                print('interface_sc_rmsd: ', row['interface_sc_rmsd'])

        row['rog'] = rog_scorefxn( pose_hal )

        DSSP.apply(pose_hal)
        ss = pose_hal.secstruct()

        row['len'] = len(pose_hal.sequence())
        row['seq'] = pose_hal.sequence()
        row['net_charge'] = row['seq'].count('K')+row['seq'].count('R')\
                     -row['seq'].count('D')-row['seq'].count('E')
        row['ss'] = ss
        row['topology'] = get_topology(ss)
        row['ss_strand_frac'] = ss.count('E') / row['len']
        row['ss_helix_frac'] = ss.count('H') / row['len']
        row['ss_loop_frac'] = ss.count('L') / row['len']

        records.append(row)

    df = pd.DataFrame.from_records(records)

    print(f'Outputting computed metrics to {outfile}')
    df.to_csv(outfile)

if __name__ == "__main__":
    main()
