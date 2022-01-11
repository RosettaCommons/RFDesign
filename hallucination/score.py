#!/software/conda/envs/SE3/bin/python

import os,sys,glob
script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)
sys.path.insert(0, script_dir+'/util')
sys.path.insert(0, script_dir+'/se3-transformer-public')

# suppress some pytorch warnings when using a cpu
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

import json
import time
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from parsers import parse_a3m,parse_pdb
from metrics import lDDT,RMSD,KL
from geometry import xyz_to_c6d,c6d_to_bins2
from trFold import TRFold
from models.ensemble import EnsembleNet
from util import num2aa

import argparse
import sys
from distutils.util import strtobool
import optimization

C6D_KEYS = ['dist','omega','theta','phi']


def get_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 
    reg_models = json.load(open(script_dir+'/models/models.json'))

    parser.add_argument("--network", type=str, required=False, dest='network_name', 
                        default='rf_v00', choices=reg_models.keys(), help='network to use')
    
    parser.add_argument("--device=", type=str, required=False, dest='device', 
                        default='cuda', choices=['cpu','cuda'], help='device to use')

    parser.add_argument("--threads=", type=int, required=False, dest='nthreads', default=1,
                        help="number of cpu threads to use for device='cpu'")

    #parser.add_argument("-f", "--fasta-dir=", type=str, required=False, dest='fas_dir', 
    #                    default='', help='folder storing protein sequences in FASTA format')

    parser.add_argument("-m", "--msa-dir=", type=str, required=False, dest='msa_dir', 
                        help='folder storing MSAs (or sequences) in fasta format (.a3m or .fas)')

    parser.add_argument("-p", "--pdb-dir=", type=str, required=False, dest='pdb_dir', 
                        help='folder storing protein structures in PDB format')

    parser.add_argument("-l", "--list=", type=str, required=False, dest='list', 
                        help='list of PDBs or FASTA-PDB pairs subject to evaluation')

    parser.add_argument("--ocsv=", type=str, required=False, dest='ocsv', 
                        help='save scoring output to a CSV files')

    parser.add_argument("--opdb=", type=str, required=False, dest='opdb', 
                        help='folder for saving PDB files')

    parser.add_argument("--onpz=", type=str, required=False, dest='onpz', 
                        help='folder for saving NPZ files')

    parser.add_argument('--weights_dir', type=str, default='/projects/ml/trDesign',
                        help='folder containing structure-prediction model weights')       
    # trFold parameters
    parser.add_argument("--use_trf", type=strtobool, required=False, default=True, help='Use TRFold to refine structure')
    parser.add_argument("-b", "--batch=", type=int, required=False, dest='batch', default=64, help='trFold batch size')
    parser.add_argument("-lr", "--learning-rate=", type=float, required=False, dest='lr', default=0.2, help='trFold learning rate')
    parser.add_argument("-n", "--nsteps=", type=int, required=False, dest='nsteps', default=100, help='trFold number of minimization steps')

    args = parser.parse_args()

    if args.list is None and args.pdb_dir is None and args.msa_dir is None:
        sys.exit('ERROR: One of --list, --pdb-dir, --msa-dir must be provided.')

    return args


def write_pdb(filename, bb, seq):
    '''save backbone in PDB format'''
    #atoms = bb.cpu().detach().numpy()
    f = open(filename,"w")
    L = bb.shape[1]
    ctr = 1
    for i in range(L):
        for j,atm_j in enumerate([" N  "," CA "," C  "]):
            f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                "ATOM", ctr, atm_j, num2aa[seq[i]],
                "A", i+1, bb[j,i,0], bb[j,i,1], bb[j,i,2],
                1.0, 0.0 ) )
            ctr += 1
    f.write("TER\n")
    f.close()


def main():
    
    separator = '# ' + '-'*78
    
    ########################################################
    # 0. process inputs
    ########################################################

    # read in info on the registered models and parse arguments
    args = get_args()
    torch.set_num_threads(args.nthreads)

    device = 0 
    Net, net_params = optimization.load_structure_predictor(script_dir, args, device)
    Net = Net.to(args.device)
    Net.eval()
    print(separator)
    print("# number of params in %s : %d"%(args.network_name,sum(p.numel() for p in Net.parameters())))

    
    ########################################################
    # 2. load sequences and structures
    ########################################################

    inputs = []

    if args.list is not None:
        print('Scoring MSA, PDB pairs from input --list')
        filenames = [line.strip().split() for line in open(args.list).readlines()]

    elif args.pdb_dir is not None and args.msa_dir is not None:
        print('Scoring MSA, PDB pairs from inputs --pdb-folder, --msa-folder')
        filenames = []
        for fn in glob.glob(os.path.join(args.pdb_dir,'*.pdb')):
            fasname = fn.replace('.pdb','.fas')
            if not os.path.exists(fasname):
                fasname = fn.replace('.pdb','.a3m')
            filenames.append([fasname,fn])

    elif args.msa_dir is not None:
        print('Predicting from MSAs/sequences from input --msa-folder')
        filenames = [[fn] for ext in ['fa','fas','fasta','a3m']
                          for fn in glob.glob(os.path.join(args.msa_dir,'*.'+ext))]

    elif args.pdb_dir is not None:
        print('Scoring PDBs from input --pdb-folder (no MSAs given)')
        filenames = [[fn] for fn in glob.glob(os.path.join(args.pdb_dir,'*.pdb'))]

    for fn in filenames:

        # msa & pdb files both provided
        if len(fn) == 2: 
            fas,pdb = fn
            item = ['.'.join(os.path.basename(fas).split('.')[:-1]),
                    '.'.join(os.path.basename(pdb).split('.')[:-1])]
            fas = parse_a3m(fas)
            pdb = parse_pdb(pdb)
            item += [torch.from_numpy(fas['msa'][None].astype(np.int64)),
                     torch.from_numpy(pdb['xyz'][:,:3].astype(np.float32)).permute(1,0,2),
                     torch.from_numpy(pdb['idx']-1)]

        # no msa file, use seq from pdb file
        elif args.pdb_dir is not None: 
            pdb = fn[0] 
            item = ['.'.join(os.path.basename(pdb).split('.')[:-1])]*2
            pdb = parse_pdb(pdb)
            item += [torch.from_numpy(pdb['seq'][None,None].astype(np.int64)),
                     torch.from_numpy(pdb['xyz'][:,:3].astype(np.float32)).permute(1,0,2),
                     torch.arange(len(pdb['idx']))] # ignore pdb residue numbering

        # no pdb file
        elif args.msa_dir is not None: 
            fas = fn[0] 
            item = ['.'.join(os.path.basename(fas).split('.')[:-1])]*2
            fas = parse_a3m(fas)
            item += [torch.from_numpy(fas['msa'][None].astype(np.int64)),
                     None, # No coordinates
                     torch.arange(fas['msa'].shape[1])] # assumes no gaps/lowercase in top seq

        inputs.append(item)
        

    ########################################################
    # 3. load background distributions
    ########################################################

    BKGDIR = "/projects/ml/trDesign/backgrounds/generic/%s.npz"
    seq_len = set([i[2].shape[-1] for i in inputs])
    bkg = {l:np.load(BKGDIR%(l)) for l in seq_len}
    bkg_nbytes = [[v2.nbytes for _,v2 in v.items()] for _,v in bkg.items()]
    bkg = {k:[torch.from_numpy(v[k2]).permute(2,0,1) for k2 in C6D_KEYS] for k,v in bkg.items()}
    print("# number of background distributions : %d (%.2fmb)"%(len(seq_len),np.sum(bkg_nbytes)/1024**2))
    

    ########################################################
    # 4. evaluate parsed proteins
    ########################################################

    print(separator)
    scoring = args.list is not None or args.pdb_dir is not None
    if scoring:
        print("# %13s %8s %8s %8s %8s | %6s"%('label','KL','CCE','Ca-lDDT','RMSD','time'))
    else:
        print("# %13s | %6s"%('label','time'))

    if args.ocsv is not None: 
        csvfile = open(args.ocsv, 'w')
        print(f'name,kl,cce,lddt,rmsd,time', file=csvfile)

    for i,inp in enumerate(inputs):
        
        tic = time.time()
        
        idx = inp[4].to(args.device)
        L = inp[2].shape[-1]
        
        ### predict ###
        with torch.no_grad():
            msa = inp[2].to(args.device)
            out = Net(msa)
        
        logits = [out[key].detach() for key in C6D_KEYS]
        probs = [nn.functional.softmax(l[0], dim=0) for l in logits]

        bkg_i = [p.to(args.device) for p in bkg[L]]
        kl = KL(probs,bkg_i).cpu().detach().numpy()
        
        if args.use_trf:
            TRF = TRFold(probs)
            if 'xyz' in out.keys():
                xyz = TRF.fold(ca_trace=out['xyz'][0,:,1,:], 
                               batch=args.batch,lr=args.lr,nsteps=args.nsteps)
            elif 'lognorm' in out.keys():
                xyz = TRF.fold(lognorm=out['lognorm'],
                               batch=args.batch,lr=args.lr,nsteps=args.nsteps)
            else:
                sys.exit('TRFold cannot generate an initial structure.')
        
        ### score (only if pdb was provided) ###
        if scoring:
            c6d = xyz_to_c6d(xyz[None],{'DMAX':20.0})
            c6d_bins = c6d_to_bins2(c6d,{'DMIN':2.0,'DMAX':20.0,'ABINS':36,'DBINS':36})
            c6d_1hot = [F.one_hot(a.squeeze(),n).permute(2,0,1) 
                        for a,n in zip(torch.split(c6d_bins[0],1,dim=-1),(37,37,37,19))]
            
            # cce between neural net prediction and TRFold output
            # NOT between neural net prediction and input structure!
            cce = [(torch.log(p+1e-6)*x-torch.log(q+1e-6)*x).sum(0).mean() for p,q,x in zip(probs,bkg_i,c6d_1hot)]
            cce = torch.stack(cce).mean().cpu().detach().numpy()

            xyz0 = inp[3].to(args.device)
            lddt = lDDT(xyz0[1],xyz[1,idx]).cpu().detach().numpy()
            rmsd = RMSD(xyz0[1],xyz[1,idx]).cpu().detach().numpy()
            
        toc = time.time()

        # catch undeclared variables. Set to None        
        if 'kl' not in locals(): kl=None
        if 'cce' not in locals(): cce=None
        if 'lddt' not in locals(): lddt=None
        if 'rmsd' not in locals(): rmsd=None
        
        if scoring:
            print("%15s %8.5f %8.5f %8.5f %8.4f | %6.3fs"%(inp[0],kl,cce,lddt,rmsd,toc-tic))
        else:
            print("%15s | %6.3fs"%(inp[0],toc-tic))
        sys.stdout.flush()

        if args.ocsv is not None:
            print(f'{inp[0]},{kl},{cce},{lddt},{rmsd},{toc-tic}', file = csvfile)
        
        if args.opdb is not None:
            if not args.use_trf:
                xyz = out['xyz'][0].permute(1,0,2)
            write_pdb("%s/%s.pdb"%(args.opdb,inp[0]),
                      xyz.cpu().detach().numpy(),
                      inp[2][0,0])

        if args.onpz is not None:
            c6d = [np.roll(p.permute(1,2,0).cpu().numpy().astype(np.float32),1,axis=-1) for p in probs]
            np.savez_compressed("%s/%s.npz"%(args.onpz,inp[0]),
                                dist=c6d[0],
                                omega=c6d[1],
                                theta=c6d[2],
                                phi=c6d[3])

    if args.ocsv is not None: csvfile.close()
    
if __name__ == '__main__':
    main()
