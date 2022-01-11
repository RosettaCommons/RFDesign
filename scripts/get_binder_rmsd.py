#!/software/conda/envs/SE3/bin/python
# 
# Calculates RMSDs between RoseTTAFold and AF2 models of a binder, after
# aligning on receptor.
#
# Usage:
#
#     ./get_binder_rmsd_af2.py FOLDER
#
# where FOLDER contains RoseTTAFold .pdbs, and FOLDER/af2/ contains AF2 pdbs
#
# Updated 2021-11-15

import pandas as pd
import numpy as np
import sys, os, argparse, glob
from collections import OrderedDict

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir+'/../util/')
import parsers

p = argparse.ArgumentParser()
p.add_argument('data_dir', help='Folder of TrDesign outputs to process')
p.add_argument('-o','--out', help='Output filename.')
p.add_argument('--template', help='Reference PDB.')
p.add_argument('--af2_dir', default='relax/', help='Subfolder with AF2 predictions.')
p.add_argument('--af2_suffix', default='_af2pred_0001', help='Filename suffix of AF2 predictions.')
args = p.parse_args()

if args.out is None:
    args.out = os.path.join(args.data_dir, 'binder_rmsd.csv')

def calc_rmsd(xyz1, xyz2, eps=1e-6):

    # center to CA centroid
    xyz1 = xyz1 - xyz1.mean(0)
    xyz2 = xyz2 - xyz2.mean(0)

    # Computation of the covariance matrix
    C = xyz2.T @ xyz1

    # Compute optimal rotation matrix using SVD
    V, S, W = np.linalg.svd(C)

    # get sign to ensure right-handedness
    d = np.ones([3,3])
    d[:,-1] = np.sign(np.linalg.det(V)*np.linalg.det(W))

    # Rotation matrix U
    U = (d*V) @ W

    # Rotate xyz2
    xyz2_ = xyz2 @ U

    L = xyz2_.shape[0]
    rmsd = np.sqrt(np.sum((xyz2_-xyz1)*(xyz2_-xyz1), axis=(0,1)) / L + eps)

    return rmsd, U

def get_binder_rmsd(fn1, fn2):
    """Align on receptor, report binder RMSD"""
    pdb_rf = parsers.parse_pdb(fn1)
    xyz_rf = pdb_rf['xyz'][:,:3]

    pdb_af2 = parsers.parse_pdb(fn2)
    xyz_af2 = pdb_af2['xyz'][:,:3]

    i_bin = np.array([i for i,x in enumerate(pdb_rf['pdb_idx']) if x[0]=='A'])
    i_rec = np.array([i for i,x in enumerate(pdb_rf['pdb_idx']) if x[0]=='B'])

    xyz_rf_rec = xyz_rf[i_rec]
    xyz_af2_rec = xyz_af2[i_rec]

    rmsd, U = calc_rmsd(xyz_rf_rec.reshape(-1,3), xyz_af2_rec.reshape(-1,3))

    xyz_rf = xyz_rf - xyz_rf_rec.reshape(-1,3).mean(0)
    xyz_af2 = xyz_af2 - xyz_af2_rec.reshape(-1,3).mean(0)

    xyz_af2 = xyz_af2 @ U

    x1 = xyz_rf[i_bin].reshape(-1,3)
    x2 = xyz_af2[i_bin].reshape(-1,3)

    L = len(i_bin)

    # binder, after aligning on receptor
    rmsd = np.sqrt(np.sum((x1-x2)**2)/(L*3))

    # whole complex
    rmsd2, U = calc_rmsd(xyz_rf.reshape(-1,3), xyz_af2.reshape(-1,3))

    return rmsd, rmsd2

def main():

    print(f'Calculating RMSDs')

    records = []
    for fn in sorted(glob.glob(args.data_dir+'/*pdb')):
        print(fn)
        pre = os.path.basename(fn).replace('.pdb','')
        fn_af2 = os.path.join(os.path.dirname(fn), args.af2_dir, pre + args.af2_suffix+'.pdb')
        if os.path.exists(fn_af2):
            rmsd, rmsd2 = get_binder_rmsd(fn, fn_af2)
        else:
            print(f'Skipping {fn} because no AF2 predictions found')
            rmsd, rmsd2 = None, None

        if args.template is not None:
            rmsd3, rmsd4 = get_binder_rmsd(fn, args.template)
        else:
            rmsd3, rmsd4 = None, None

        records.append(dict(
            name=pre,
            binder_rmsd_af2_des=rmsd,
            complex_rmsd_af2_des=rmsd2,
            binder_rmsd=rmsd3,
            complex_rmsd=rmsd4
        ))
    df = pd.DataFrame.from_records(records)

    print(f'Outputting computed metrics to {args.out}')
    df.to_csv(args.out)

if __name__ == "__main__":
    main()
