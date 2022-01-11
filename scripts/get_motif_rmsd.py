#!/software/conda/envs/SE3/bin/python
# 
# Calculates motif RMSDs between design models and reference structure for a
# folder of PDBs and outputs to a CSV. 
#
# Usually you should use pyrosetta_metrics.py instead of this, as that also
# does a few other useful metrics. But this script is much faster for RMSDs,
# which is helpful in certain situations.
#
# This script ONLY does backbone RMSD and does NOT handle contig sets.
#
# Usage:
#
#     ./get_motif_rmsd.py DESIGNS_FOLDER
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
# Updated 2021-10-13

import pandas as pd
import numpy as np
import sys, os, argparse, glob
from collections import OrderedDict
from icecream import ic

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir+'/../util/')
import parsers

p = argparse.ArgumentParser()
p.add_argument('data_dir', help='Folder of TrDesign outputs to process')
p.add_argument('-t','--template', help='Template (natural binder) structure (.pdb)')
#p.add_argument('--sc_rmsd', action='store_true', default=True, 
#    help='Also calculate side-chain RMSD, returning NaN if residues aren\'t matched.')
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

def main():

    # input and output names
    if args.out is not None:
        outfile = args.out if '.csv' in args.out else args.out+'.csv'
    else:
        outfile = os.path.join(args.data_dir,'motif_rmsd.csv')

    if args.template is not None:
        pdb_ref = parsers.parse_pdb(args.template)
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

        if args.template is None and 'settings' in trb and trb['settings']['pdb'] != last_template:
            pdb_ref = parsers.parse_pdb(trb['settings']['pdb'])
            last_template = trb['settings']['pdb']

        pdb_hal = parsers.parse_pdb(fn)

        # BB RMSD
        idxmap = dict(zip(pdb_ref['pdb_idx'],range(len(pdb_ref['pdb_idx']))))
        ref_idx0 = [idxmap[idx] for idx in trb['con_ref_pdb_idx']]
        xyz_motif_ref = pdb_ref['xyz'][ref_idx0,:3].reshape(-1,3)
        xyz_motif_hal = pdb_hal['xyz'][trb['con_hal_idx0'],:3].reshape(-1,3)

        row['contig_rmsd'] = calc_rmsd(xyz_motif_ref, xyz_motif_hal)
        print('contig_rmsd: ', row['contig_rmsd'])

        if args.interface_res is not None:
            int_res_pdb_idx = expand(args.interface_res)

            idxmap = dict(zip(trb['con_ref_pdb_idx'],trb['con_ref_idx0']))
            int_idx_ref = [idxmap[pdb_idx] for pdb_idx in int_res_pdb_idx]
            idxmap2 = dict(zip(trb['con_ref_pdb_idx'],trb['con_hal_idx0']))
            int_idx_hal = [idxmap2[pdb_idx] for pdb_idx in int_res_pdb_idx]

            xyz_int_ref = pdb_ref['xyz'][int_idx_ref,:3].reshape(-1,3)
            xyz_int_hal = pdb_hal['xyz'][int_idx_hal,:3].reshape(-1,3)
            
            row['interface_rmsd'] = calc_rmsd(xyz_int_ref, xyz_int_hal)
            print('interface_rmsd: ', row['interface_rmsd'])

        records.append(row)

    df = pd.DataFrame.from_records(records)

    print(f'Outputting computed metrics to {outfile}')
    df.to_csv(outfile)

if __name__ == "__main__":
    main()
