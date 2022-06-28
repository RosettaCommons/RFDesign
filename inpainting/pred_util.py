import sys, os, copy
import torch 
import numpy as np
from icecream import ic
from inpaint_util import ResidueMap
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
            Parsed command-line arguments, in particular the variables: window, contigs, receptor_chain.
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
    if args.contigs:        
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

def get_residue_map(args, parsed_pdb):
    if 'len' in args and args.len:
        assert len(args.len) == len(args.contigs), 'If passing lengths, the number of entries must match the number of contig entries. Use 0 for contigs that sample gap lengths (instead of scattering contigs).'
    else:
        args.len = len(args.contigs) * [0]
    
    contig_list = []
    for L_range, con in zip(args.len, args.contigs):
        _, mappings = contigs.apply_mask(con, parsed_pdb)
            
        contig_list.append(mappings['sampled_mask'])
            
        # Add any inpainting regions
        #inpaint_seq_ranges = [('ref', sel) for sel in args.inpaint_seq.split(',')] if args.inpaint_seq else []
        inpaint_seq_ranges = [('ref', sel) for sel in args.inpaint_seq] if args.inpaint_seq else []
        #inpaint_str_ranges = [('ref', sel) for sel in args.inpaint_str.split(',')] if args.inpaint_str else []
        inpaint_str_ranges = [('ref', sel) for sel in args.inpaint_str] if args.inpaint_str else []
        
    print('asdfasf', contig_list)
    rm = ResidueMap(contig_list=contig_list, 
                    ref_pdb_idxs=parsed_pdb['pdb_idx'],
                    inpaint_seq_ranges=inpaint_seq_ranges,
                    inpaint_str_ranges=inpaint_str_ranges,
                   )
    mappings = rm.mappings

    return rm, mappings 
  
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
