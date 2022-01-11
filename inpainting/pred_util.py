# Helper module for creating masks and masked tensors for RFold input 

import torch 
import numpy as np
#from icecream import ic
from dj_util import SampledMask

import sys, os, copy

# absolute path to folder containing this file
script_dir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(script_dir+'/../hallucination/util/')
import contigs


# hyperparams for model 
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


def get_mask(args, parsed_pdb, chain_idx_gap=500):
    """
    Gets 1-d boolean mask corresponding to residue positions to replace/inpaint.
    
    Parameters
    ----------
        args : argparse.Namespace
            Parsed command-line arguments, in particular the variables: window, contigs, len, keep_order, min_gap, receptor_chain.
        parsed_pdb : dict
            Output of parse_pdb command, in particular the key pdb_idx.
            
    Returns
    -------
        mask : np.array of bool (L,)
            Boolean array of length L of designed protein indicating positions to be replaced/inpainted.
        mappings : dict
            Dictionary with various indices describing the mapping of kept regions:
                con_ref_pdb_idx: kept regions, original pdb numbering [(chain, num),...]
                con_ref_idx0:    kept regions, 0-indexed continguous "pytorch tensor numbering"
                con_hal_pdb_idx: kept regions, output pdb numbering, 1-indexed contiguous by chain [(chain, num),...]
                con_hal_idx0:    kept regions, 0-indexed output numbering
                idx_rf:          indices for RoseTTAFold input (with large offsets added between chains)
                hal_idx1:        all regions (masked and kept), output pdb numbering, 1-indexed contiguous by chain
    """
    # contig/inverse mask mode: specify residues to keep
    if args.contigs is not None:
        # contig mode: chain letter starting each range
        if all([x[0].isalpha() for x in args.contigs.split(',')]): 
            if args.len is None:
                sys.exit('ERROR: Either --len must be given or --contigs argument must have gap lengths.')
            _, mappings = contigs.scatter_contigs(contigs=args.contigs, 
                                                  pdb_out=parsed_pdb, 
                                                  L_range=args.len, 
                                                  keep_order=args.keep_order, 
                                                  min_gap=args.min_gap)

        # inverse mask mode: some ranges don't have chain letter
        else: 
            _, mappings = contigs.apply_mask(args.contigs, parsed_pdb)
            
        mask_str = mappings['sampled_mask']
        sm = SampledMask(mask_str, ref_pdb_idxs=parsed_pdb['pdb_idx'])
        
        # add a receptor
        if args.receptor_chain:
            pdb_idx_rec = [x for x in parsed_pdb['pdb_idx'] if x[0]==args.receptor_chain]
            rec_str = SampledMask.contract(pdb_idx_rec)
            sm.add_receptor(rec_str, location='second')
     
    else:
        # add entire protein
        # assumes masking only happens on chain A
        pdb_idx = parsed_pdb['pdb_idx']
        pdb_idx_A = [idx for idx in pdb_idx if idx[0]=='A']
        mask_str = SampledMask.contract(pdb_idx_A)
        sm = SampledMask(mask_str, ref_pdb_idxs=pdb_idx_A)
        
        # window mode: specify residues to replace using DJ's syntax ("A,1,10:A,15,15")
        if args.window:
            window = args.window.split(':')           # list of A,B where A is start, B is end 
            window = [i.split(',') for i in window]   # list of lists. [[chain1, a1, b1],[chain2, a2,b2]]
            window = [[ch,int(r1), int(r2)] for ch,r1,r2 in window]
            for ch, s, e in window:
                sm.add_inpaint_range(f'{ch}{s}-{e}')

        # mask mode: specify residues to replace using Doug/Jue syntax ("A1-10,A15")
        elif args.mask:
            for region in args.mask.split(','):
                sm.add_inpaint_range(region)

        # add a receptor
        if args.receptor_chain:
            pdb_idx_rec = [x for x in parsed_pdb['pdb_idx'] if x[0]==args.receptor_chain]
            rec_str = SampledMask.contract(pdb_idx_rec)
            sm.add_receptor(rec_str, location='second')

    mask_str = sm.inpaint
    mappings = sm.mappings
    
    # mask sequence separately from structure
    if args.mask_seq:
        sm_seq = copy.deepcopy(sm)
        for region in args.mask_seq.split(','):
            sm_seq.add_inpaint_range(region)
    else:
        sm_seq = sm

    mask_seq = sm_seq.inpaint

    return sm, mappings, mask_str, mask_seq


def mask_inputs(args, mask_str, seq, msa, msa_full, xyz_t, f1d_t, mask_seq=None):

    if mask_seq is None:
        mask_seq = mask_str

    if args.task == 'den':
        # do denoising preprocessing 
        # Ideally, here we would do mutation of amino acids and noising of structure, etc
        pass 

    elif args.task == 'des':

        # remove sequence information within the window (seq --> 20, msa --> 21)
        msa[:,mask_seq]      = args.msa_mask_token
        msa_full[:,mask_seq] = args.msa_mask_token
        seq[mask_seq]      = 20
        if args.mask_query:
            seq[:] = 20
            msa[0,:] = args.msa_mask_token
            msa_full[0,:] = args.msa_mask_token

    elif args.task == 'str':
        # do structure prediction preprocessing
        xyz_t[:,:,mask_str] = np.nan

    elif args.task == 'hal':
        # do hallucination preprocessing (combination of des + str) 

        if args.init_seq == None:
            msa[:,mask_seq]      = args.msa_mask_token
            msa_full[:,mask_seq] = args.msa_mask_token
            seq[mask_seq]        = 20

        elif args.init_seq == 'poly':
            # poly AA initialization
            msa[:,mask_seq]      = util.aa2num[args.poly]
            msa_full[:,mask_seq] = util.aa2num[args.poly]
            seq[mask_seq]        = util.aa2num[args.poly]

        else:
            raise NotImplementedError('Shouldnt be here :/')

        f1d_t[:,:,mask_str] = 0 # zero confidence in the masked region(s)

        # remove structure information within the window (change template coords to nan)
        xyz_t[:,:,mask_str] = float('nan')

    else:
        sys.exit(f'Invalid task {args.task}')
    
    return seq, msa, msa_full, xyz_t, f1d_t 

def rw_ins_input(args,mask_orig,xyz_t,seq,msa,msa_full,mask,DEVICE):
    idx_ins      = np.random.choice(np.where(mask_orig)[0], 1, replace=False)[0]
    xyz_t_new    = torch.full((1,xyz_t.shape[1],1,3,3),np.nan).to(DEVICE) #torch.full((1,1,L,3,3),np.nan)
    xyz_t        = torch.cat([xyz_t[:,:,:idx_ins], xyz_t_new, xyz_t[:,:,idx_ins:]], 2) 
    seq_new      = torch.full((1,), 20)[None].to(DEVICE) 
    seq          = torch.cat([seq[:,:idx_ins],seq_new,seq[:,idx_ins:]],1) 
    msa_new      = torch.full((msa.shape[0],1), args.msa_mask_token)[None].to(DEVICE) 
    msa          = torch.cat([msa[:,:,:idx_ins],msa_new,msa[:,:,idx_ins:]],2)
    msa_full_new = torch.full((msa_full.shape[0],1), args.msa_mask_token)[None].to(DEVICE) 
    msa_full     = torch.cat([msa_full[:,:,:idx_ins],msa_full_new,msa_full[:,:,idx_ins:]],2)
    mask_orig    = np.insert(mask_orig, idx_ins, True, axis=0)
    mask         = np.insert(mask, idx_ins, True, axis=0)
    return mask_orig,xyz_t,seq,msa,msa_full,mask,idx_ins

def rw_del_input(args,mask_orig,xyz_t,seq,msa,msa_full,mask,DEVICE):
    idx_del      = np.random.choice(np.where(mask_orig)[0], 1, replace=False)[0]
    xyz_t        = torch.cat([xyz_t[:,:,:idx_del], xyz_t[:,:,idx_del+1:]], 2)  
    seq          = torch.cat([seq[:,:idx_del],seq[:,idx_del+1:]],1) 
    msa          = torch.cat([msa[:,:,:idx_del],msa[:,:,idx_del+1:]],2)
    msa_full     = torch.cat([msa_full[:,:,:idx_del],msa_full[:,:,idx_del+1:]],2)
    mask_orig    = np.delete(mask_orig, idx_del, axis=0)
    mask         = np.delete(mask, idx_del, axis=0)
    return mask_orig,xyz_t,seq,msa,msa_full,mask,idx_del
