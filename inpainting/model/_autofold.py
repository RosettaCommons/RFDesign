import sys, os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from functools import partial
import argparse 

# from minkyung 
import data_loader 
from data_loader import get_train_valid_set,\
                        loader_tbm, Dataset,\
                        tbm_collate_fn, loader_fixbb,\
                        collate_fixbb, loader_mix, collate_mix,\
                        DeNovoDataset, loader_denovo, collate_denovo

from kinematics import xyz_to_c6d, c6d_to_bins2, xyz_to_t2d, xyz_to_bbtor
from TrunkModel  import TrunkModule
from parsers import parse_a3m, parse_pdb 
from loss import *
import util 
from scheduler import get_linear_schedule_with_warmup, CosineAnnealingWarmupRestarts

# dj external functions 
import dj_util
import dj_parser
import prediction_methods 

from icecream import ic 

os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # debbugging


torch.set_printoptions(sci_mode=False)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    
    # model params from Minkyung's pretrained checkpoint 

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--checkpoint', '-ckpt',    type=str,   default='/home/davidcj/projects/BFF/design/dj_design/from_minkyung_072021/models/BFF_last.pt',  help='Model checkpoint to load')
    parser.add_argument('--pdb',        '-pdb',     type=str,   default=None,                                               help='input pdb file to operate on')                                            
    parser.add_argument('--pt_file',    '-pt',      type=str,   default=None,                                               help='input .pt file to load and operate on (instead of parsing a pdb)')
    parser.add_argument('--outf',       '-outf',    type=str,   default='./pdbs_test',                                      help='directory to spill output into')
    parser.add_argument('--dump_pdb',   '-dump_pdb',            default=False,          action='store_true',                help='dump pdb files into output directories?')
    
    # general modelling args 
    parser.add_argument('--task',       '-task',    type=str,   default='des',      choices=['des', 'str', 'hal', 'den'],   help='Which prediction mode')
    parser.add_argument('--window',     '-w',       type=str,   default='1,2',                                              help='Beinning and end of prediction window (indlusive, exclusive)')

    # sequecne design args 
    parser.add_argument('--inf_method',             type=str,   default='one_shot',   choices=['one_shot',
                                                                                             'multi_shot',
                                                                                             'autoregressive',
                                                                                             'confidence',
                                                                                             'gibbs'],                      help='Which sequence design method to use')
    parser.add_argument('--des_conf_topk',          type=int,   default=2,                                                  help='If using design by confidence mode, how many residues to design at once?')
    parser.add_argument('--n_iters',                type=int,   default=2,                                                  help='If using multi shot, how many iters to use')
    parser.add_argument('--slide',                              default=False,      action='store_true',                    help='If true, makes predictions for all possible "reading frames" along the sequence')
    parser.add_argument('--init_seq',               type=str,   default=None,       choices=[None, 'poly'],                 help='choose sequence initialization method')
    parser.add_argument('--poly',                    type=str,   default='ALA',                                             help='3-letter code of amino acid to initialize unknown sequence with') 
    parser.add_argument('--clamp_1d',               type=float,  default=None,                                              help='Clamp recycled 1D feature inputs to certain value')

    args = parser.parse_args()

    # args processing and sanity checks here 
    args.window = args.window.split(':')                # list of A,B where A is start, B is end 
    args.window = [i.split(',') for i in args.window]   # list of lists. [[a1, b1],[a2,b2]]
    args.window = [[int(j) for j in i] for i in args.window]

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    
    
    # model params from Minkyung's pretrained checkpoint 
    model_param = {'SE3_param': {'div': 2,
                                'l0_in_features': 32,
                                'l0_out_features': 8,
                                'l1_in_features': 3,
                                'l1_out_features': 2,
                                'n_heads': 4,
                                'num_channels': 16,
                                'num_degrees': 2,
                                'num_edge_features': 32,
                                'num_layers': 3},
                  'd_hidden': 64,
                  'd_msa': 384,
                  'd_msa_full': 64,
                  'd_pair': 288,
                  'd_templ': 64,
                  'n_head_msa': 12,
                  'n_head_pair': 8,
                  'n_head_templ': 4,
                  'n_layer': 1,
                  'n_module': 8,
                  'n_module_str': 4,
                  'n_resblock': 1,
                  'p_drop': 0.15,
                  'performer_L_opts': {'feature_redraw_interval': 10000, 'nb_features': 64},
                  'performer_N_opts': {'feature_redraw_interval': 10000, 'nb_features': 64},
                  'r_ff': 4}
    model_param['use_templ'] = True

    # make model 
    model = TrunkModule(**model_param).to(DEVICE)
    model.eval()


    # load checkpoint into model 
    print('Loading checkpoint')
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    print('Successfully loaded pretrained weights into model')

    # cross entropy loss 
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    full_loss_s = []
    names = []
    tasks = []
    ctr = 0

    # get prediction method to use 
    if args.inf_method == 'one_shot':
        prediction_method = prediction_methods.one_shot
    elif args.inf_method == 'multi_shot':
        prediction_method = prediction_methods.multi_shot 
    else:
        raise NotImplementedError('Other prediction methods not implemented yet')
    print('Using inference method ',args.inf_method)

    ## Loading up the protein to be operated on

    parsed = dj_parser.parse_pdb(args.pdb)      # parse pdb file 
    #parsed = dj_parser.parse_pdb('/projects/ml/TrRosetta/benchmarks/denovo2021/pdb/2KL8.pdb')   # for tests, parse 2KL8 

    xyz_true, mask, idx, seq, pdb_idx = parsed['xyz'], parsed['mask'], parsed['idx'], parsed['seq'], parsed['pdb_idx']

    xyz_true    = torch.from_numpy(xyz_true)[:,:3,:] # N,CA,C
    mask        = torch.from_numpy(mask)
    idx         = torch.from_numpy(idx)
    seq         = torch.from_numpy(seq)
    seq_orig    = torch.clone(seq)

    l = seq.shape[0]

    # empty template features
    xyz_t = torch.full_like(xyz_true[:,:3,:], float('nan'))[None,None,...]
    f1d_t = torch.zeros(1,l,1)  # NOTE: Can mess with template confidences here. e.g., f1d_t[START:END] = 1
    f0d_t = torch.zeros(1,3)    # NOTE: Can mess with template confidences here 

    # replace template structure with true structure (yes, this is redundant with code above)
    xyz_t = torch.clone(xyz_true)[None,None,:,:3,:] #[1,1,L,3,3] N,Ca,C

    # make msa
    msa = seq[None,...]          # add a dimensin so its now [N,L] = [1,L]
    msa_full = torch.clone(msa)
    label_aa_s = torch.clone(msa)

    # preprocess seq/str to match task 
    
    for i,W in enumerate(args.window):
        START, END = W
        END += 1

        if args.task == 'den':
            # do denoising preprocessing
            pass 

        elif args.task == 'des':
            # do design preprocessing 

            # remove sequence information within the window (seq --> 20, msa --> 21) 
            if i == 0:
                mask_aa_s = torch.full_like(msa, False).bool()
            mask_aa_s[:,START:END] = True

            msa[mask_aa_s]      = 21
            msa_full[mask_aa_s] = 21
            seq[mask_aa_s[0]]   = 20

            # tell model the bb information is good 
            if i == 0:
                f1d_t = torch.full_like(f1d_t, 1).float()[None,...]   # NOTE: Can play with template confidences here 
                f0d_t = torch.full_like(f0d_t, 1).float()[None,...]   # NOTE: ^^ 

        elif args.task == 'str':
            # do structure prediction preprocessing
            # Make MSA mask that wont actually mask anything - just for loss calc 
            if i == 0:
                mask_aa_s = torch.full_like(msa, False).bool()
                # remove structure information within the window (change template coords to nan)
                str_mask = torch.full_like(xyz_t, False).bool()

            str_mask[:,:,START:END,:,:] = True 

            xyz_t[str_mask] = float('nan')

            # tell model bb information is good/bad? 
            if i == 0:
                f1d_t = torch.full_like(f1d_t, 1).float()[None,...]   # NOTE: Can play with template confidences here
                f0d_t = torch.full_like(f0d_t, 1).float()[None,...]   # NOTE: ^^


        elif args.task == 'hal':
            # do hallucination preprocessing 

            # (1)  do design preprocessing
            # remove sequence information within the window (seq --> 20, msa --> 21)
            if i == 0:
                mask_aa_s = torch.full_like(msa, False).bool()

            mask_aa_s[:,START:END] = True

            if args.init_seq == None:
                msa[mask_aa_s]      = 21
                msa_full[mask_aa_s] = 21
                seq[mask_aa_s[0]]   = 20

            elif args.init_seq == 'poly':
                # poly AA initialization 
                msa[mask_aa_s]      = util.aa2num[args.poly]
                msa_full[mask_aa_s] = util.aa2num[args.poly]
                seq[mask_aa_s[0]]   = util.aa2num[args.poly]

            else:
                raise 



            # tell model the bb information is good
            if i == 0:
                f1d_t = torch.full_like(f1d_t, 1).float()[None,...]   # NOTE: Can play with template confidences here
                f0d_t = torch.full_like(f0d_t, 1).float()[None,...]   # NOTE: ^^

            f1d_t[:,:,START:END,:] = 0


            # (2) do str preprocessing 

            # remove structure information within the window (change template coords to nan)
            if i == 0:
                str_mask = torch.full_like(xyz_t, False).bool()

            str_mask[:,:,START:END,:,:] = True

            xyz_t[str_mask] = float('nan')

        else:
            sys.exit(f'Invalid task {args.task}')
    
    ## Prepare true information for loss calc, make prediction(s) and calculate losses ## 
    t2d   = xyz_to_t2d(xyz_t, f0d_t)    

    c6d,_ = xyz_to_c6d(xyz_true[None,...])
    c6d   = c6d_to_bins2(c6d)
    tors  = xyz_to_bbtor(xyz_true[None,...])
    
    # proper batching / device assignment 
    msa         = msa[None,...].to(DEVICE)
    msa_full    = msa_full[None,...].to(DEVICE)
    label_aa_s  = label_aa_s[None,...].to(DEVICE)
    seq         = seq[None,...].to(DEVICE)   
    idx         = idx[None,...].to(DEVICE)
    f1d_t       = f1d_t.to(DEVICE)
    t2d         = t2d.to(DEVICE)
    
    item = {
            'msa'       : msa,
            'msa_full'  : msa_full, 
            'seq'       : seq, 
            'idx'       : idx, 
            't1d'       : f1d_t, 
            't2d'       : t2d,
            'w_start'   : START,
            'w_end'     : END,
            't0d'       : f0d_t,
            'xyz_true'  : xyz_true, 
            'device'    : DEVICE, 
            'args'      : args}
    #logit_s, logit_aa_s, logit_tor_s, pred_crds, pred_lddts = model(msa, 
    #                                                                msa_full, 
    #                                                                seq, 
    #                                                                idx, 
    #                                                                t1d=f1d_t, 
    #                                                                t2d=t2d,
    #                                                                use_transf_checkpoint=True)
    logit_s, logit_aa_s, logit_tor_s, pred_crds, pred_lddts = prediction_method(model, item)

    
    ## Measure performance at the regions where we wanted a good prediction ##  
    B,L = xyz_true.shape[:2]
    
    # Sequence recovery 
    TOP1,TOP2,TOP3 = 1,2,3
    logits_argsort = torch.argsort(logit_aa_s, dim=1, descending=True)
    acc1 = (label_aa_s[...,mask_aa_s] == logits_argsort[:, :TOP1, mask_aa_s.reshape(B,-1).squeeze()]).any(dim=1)
    acc2 = (label_aa_s[...,mask_aa_s] == logits_argsort[:, :TOP2, mask_aa_s.reshape(B,-1).squeeze()]).any(dim=1)
    acc3 = (label_aa_s[...,mask_aa_s] == logits_argsort[:, :TOP3, mask_aa_s.reshape(B,-1).squeeze()]).any(dim=1)

    ic(acc1)
    ic(acc1.shape)
    ic(acc1.float().mean())
    ic(acc2.float().mean())
    ic(acc3.float().mean())


        
    # Output pdb if desired 
    if args.dump_pdb:
        print('WARNING HAVENT CHANGED SEQ YET')
        print('OUTPUT PDB HAS ORIGINAL SEQ')
        #util.writepdb(os.path.join(args.outf, args.task) + '_pred.pdb', pred_crds[-1].squeeze(), torch.ones(pred_crds.shape[2]), seq_orig.squeeze())
        util.writepdb(os.path.join(args.outf, args.task) + '_pred.pdb', pred_crds[-1].squeeze(), mask_aa_s.long().squeeze(), seq_orig.squeeze())


        


if __name__ == '__main__':
    main()
