import sys, os, subprocess, pickle, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils import data
from functools import partial
import argparse
import random
import copy
from collections import namedtuple
from icecream import ic
import math

# inpainting associated things
import inf_methods
from inpaint_util import get_translated_coords, get_tied_translated_coords, translate_coords, parse_block_rotate, rotate_block, write_pdb
import pred_util
from contigs import ContigMap, get_mappings
import parsers

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir+'/model')
from RoseTTAFoldModel import RoseTTAFoldModule
from mask_generator import generate_masks 
from data_loader import MSAFeaturize_fixbb, TemplFeaturizeFixbb 
from kinematics import xyz_to_t2d
import util

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

MODEL_PARAM = {'SE3_param': {'div': 4,
                                'l0_in_features': 32,
                                'l0_out_features': 32,
                                'l1_in_features': 3,
                                'l1_out_features': 2,
                                'n_heads': 4,
                                'num_channels': 32,
                                'num_degrees': 2,
                                'num_edge_features': 32,
                                'num_layers': 3},
                  'd_hidden': 32,
                  'd_hidden_templ': 64,
                  'd_msa': 256,
                  'd_msa_full': 64,
                  'd_pair': 128,
                  'd_templ': 64,
                  'n_head_msa': 8,
                  'n_head_pair': 4,
                  'n_head_templ': 4,
                  'n_module_2track': 24,
                  'n_module_3track': 8,
                  'p_drop': 0.15}

def dump_args(args):
    """
    Dump flags into output dir 
    """
    outdir = args.outdir

    with open(os.path.join(outdir, 'FLAGS.txt'), 'w') as fp:
        for key,val in vars(args).items():
            fp.write(str(key) + ' '*8 + str(val) + '\n')


def process_args(args):
    """
    Does any argument postprocessing     
    """
    if args.dump_all:
        args.dump_pdb   = True
        args.dump_fasta = True
        args.dump_trb   = True
        args.dump_npz   = True

    if args.out != None:
        args.outdir= '/'.join(args.out.split('/')[:-1])
        args.prefix = args.out.split('/')[-1]
        
    return args 
    

def get_args():
    """
    Parse command line args
    """
    parser = argparse.ArgumentParser()
    
    # design-related args
    parser.add_argument('--pdb','-p',dest='pdb',
            help='input protein')
    parser.add_argument('--contigs', default=None, nargs='+',
            help='Pieces of input protein to keep ')
    parser.add_argument('--length',default=None,type=str,
            help='Specify length, or length range, you want the outputs. e.g. 100 or 95-105')
    parser.add_argument('--checkpoint', default=script_dir+'/weights/BFF_mix_epoch25.pt',
            help='Checkpoint to pretrained RFold module')
    parser.add_argument('--inpaint_str', type=str, default=None, nargs='+',
         help='Predict the structure at these residues. Similar mask (and window), but is specifically for structure')
    parser.add_argument('--inpaint_seq', type=str, default=None, nargs='+',
         help='Predict the sequence at these residues. Similar mask (and window), but is specifically for sequence.')
    parser.add_argument('--n_cycle', type=int, default=4, 
            help='Number of recycles through RFold')
    parser.add_argument('--tmpl_conf', type=str, default='0.5', 
            help='1D confidence value for template residues')
    parser.add_argument('--num_designs', type=int, default=1, 
            help='Number of designs to make')
    parser.add_argument('--res_translate',type=str,default=None,
        help='Which residues to translate (randomly in x, y and z direction), '\
             'with maximum distance to translate specified, e.g. "A35,2:B22,4" translates '\
             'residue A35 up to 2A in a random direction, and B22 up to 4A. If specified '\
             'residues are in masked --window, they will be unmasked. In --contig mode, residues'\
             'must not be masked (as need to know where to put them. Default distance to translate is 2A.')
    parser.add_argument('--tie_translate',type=str,default=None,
        help='For randomly translating multiple residues together (e.g. to move a whole secondary structure element). '\
             'Syntax is e.g. "A22,A27,A30,4.0:A48,A50" which '\
             'would randomly move residues A22, A27 and A30 together up to 4A, and A48 and A50 together (but in a different random direction/distance to the first block) '\
             'to a default distance of up to 2A.'\
             'Alternatively, residues can be specifed like "A12-26,6.0:A40-52,A56". '\
             'This can be specified alongside --res_translate, so some residues are tied, and some are not, but if residues are specified in both, they will only be moved in their'\
             ' tied block (i.e. their --res_translate will be ignored)')    
    parser.add_argument('--floating_points',type=str,default=None,
            help="Do you want to input some single residues as 'floating points', rather than residues. Indicated single residues are provided only as Cb-Cb distance to other atoms, with all angle information masked") 
    parser.add_argument('--block_rotate',type=str,default=None,
            help="Do you want to rotate a whole structural block (or single residue)? Syntax is same as tie_translate.")
    
    # i/o args
    parser.add_argument('--verbose', '-v', dest='verbose', default=False, action='store_true',
            help='Boolean flag for printing / debugging')
    parser.add_argument('--out', default='pdbs_test/auto_out',
            help='output directory and for files')
    parser.add_argument('--dump_pdb', default=False, action='store_true',
            help='Whether to dump pdb output')
    parser.add_argument('--dump_trb', default=False, action='store_true',
            help='Whether to dump trb files in output dir')
    parser.add_argument('--dump_npz', default=False, action='store_true',
            help='Whether to dump npz (disto/anglograms) files in output dir')
    parser.add_argument('--dump_fasta', '--dump_fas', dest='dump_fasta', default=False, action='store_true',
            help='Whether to dump fasta file in output dir')
    parser.add_argument('--dump_all', default=False, action='store_true',
            help='If true, will dump all possible outputs to outdir')
    parser.add_argument('--input_json', type=str, default=None,
        help='JSON-formatted list of dictionaries, each containing command-line arguments for 1 '\
             'design.')

     
    args = parser.parse_args()
    args = process_args(args)
     
    return args



def MSAFeaturize_fixbb_inference(seq, params):
    """
    Turn the single sequence msa into appropriate tensor inputs 

    Parameters:
        seq (torch.tensor, required): Tensor of shape [L,] of protein sequence as integers 

        params (dict, required): Dictionary of parameters 
    
    Returns:
        fill this in 
    """
    msa = seq[None,...].clone()
    N,L = msa.shape 

    b_seq = list()
    b_msa_clust = list()
    b_msa_seed = list()
    b_msa_extra = list()
    b_mask_pos = list()

    for i_cycle in range(params['MAXCYCLE']):
        
        # Make 43 dimensional MSA 
        msa_onehot = torch.nn.functional.one_hot(msa.clone(), num_classes=22)
        fake_ins   = torch.zeros_like(msa_onehot)[:,:,:2] 
        msa_seed   = torch.cat([msa_onehot, msa_onehot, fake_ins], dim=-1)
        b_msa_clust.append(msa[:1].clone())
        b_msa_seed.append(msa_seed.clone())

        # append to batch sequences 
        b_seq.append(msa_onehot[0].clone())

        # Make 23 dimensional MSA 
        msa_extra = msa_onehot.clone()
        fake_ins2 = torch.zeros_like(msa_extra)[:,:,:1]
        msa_extra = torch.cat([msa_extra, fake_ins2], dim=-1)

        b_msa_extra.append(msa_extra)

        b_mask_pos.append(torch.zeros_like(msa).bool()) # dummy variable 

    b_seq       = torch.stack(b_seq)
    b_msa_clust = torch.stack(b_msa_clust)
    b_msa_seed  = torch.stack(b_msa_seed)
    b_msa_extra = torch.stack(b_msa_extra)
    b_mask_pos  = torch.stack(b_mask_pos)

    return b_seq, b_msa_clust, b_msa_seed, b_msa_extra, b_mask_pos

 
def main():

    args = get_args()

    design_params = {'MAXLAT'           : 1,            # dummy val
                     'FIX_BB_MUT_FRAC'  : 0.0}
    
    inf_method = inf_methods.classic_inference

    # make model and load checkpoint 
    print('Loading model checkpoint...')
    model = RoseTTAFoldModule(**MODEL_PARAM).to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model_state = ckpt['model_state_dict']
    model.load_state_dict(model_state)
    print('Successfully loaded model checkpoint')
    
    # loop through dicts of arguments from json input
    if args.input_json is not None:
        with open(args.input_json) as f_json:
            argdicts = json.load(f_json)
        print(f'List of argument dicts loaded from JSON {os.path.abspath(args.input_json)}')
    else:
        # no json input, spoof list of argument dicts
        argdicts = [{}]


    for i_argdict, argdict in enumerate(argdicts):

        if args.input_json is not None:
            print(f'\nAdding argument dict {i_argdict} from input JSON ({len(argdicts)} total):')
            print(argdict)
            for k,v in argdict.items():
                setattr(args, k, v)


        # output prefix for this set of arguments
        if args.input_json is None or \
            (args.input_json is not None and \
                ('out' in argdict or 'outf' in argdict or 'prefix' in argdict)):
            # new output name specified in this argdict
            argdict_prefix = args.out
        else:
            # same output prefix, add number to make it unique
            argdict_prefix = args.out+f'_{i_argdict}'
               
        design_params['MAXCYCLE'] = args.n_cycle

        # parse pdb 
        parsed_pdb = parsers.parse_pdb(args.pdb)
        L = len(parsed_pdb['idx'])
        
        #get positions of residues to translate
        if args.res_translate is not None:
            res_translate = get_translated_coords(args)
        if args.tie_translate is not None:
            if args.res_translate is not None:
                res_translate = get_tied_translated_coords(args, res_translate)
            else:
                res_translate = get_tied_translated_coords(args)

        if args.block_rotate is not None:
            block_rotate = parse_block_rotate(args)
            print(block_rotate)

        process_args(args)
        dump_args(args)
        
        ic(args)

        for i_des in range(args.num_designs):
            print('On design ',i_des)
        
            # process contigs and generate masks
            ic(args.contigs)
            rm = ContigMap(parsed_pdb, args.contigs, args.inpaint_seq, args.inpaint_str, args.length)
            mappings = get_mappings(rm)
            mask_str = torch.from_numpy(rm.inpaint_str)[None,:]
            mask_seq = torch.from_numpy(rm.inpaint_seq)[None,:]
            
            # get raw inputs before remapping 
            if args.res_translate is not None or args.tie_translate is not None:
                xyz_true,translate_dict = translate_coords(parsed_pdb, res_translate)
                xyz_true = torch.from_numpy(xyz_true) 
            else:
                xyz_true = torch.from_numpy(parsed_pdb['xyz'][:,:3,:])
            if args.block_rotate is not None:
                xyz_true,rotate_dict = rotate_block(xyz_true,block_rotate,parsed_pdb['pdb_idx'])
            
            seq = torch.from_numpy(parsed_pdb['seq'])
            
            # scatter/map the residues according to the contigs   
            xyz_t = torch.full((1,1,len(rm.ref),3,3), np.nan)
            xyz_t[:,:,rm.hal_idx0,:,:] = xyz_true[rm.ref_idx0,:,:][None, None,...]
            seq_t = torch.full((1,len(rm.ref)),20).squeeze()
            seq_t[rm.hal_idx0] = seq[rm.ref_idx0]
            seq=seq_t
            # template confidence
            conf_1d = torch.ones_like(seq)*float(args.tmpl_conf)
            conf_1d[~mask_str[0]] = 0 # zero confidence for places where structure is masked

            # mask sequence and structure
            seq[~mask_seq[0]] = 20 
            xyz_t[:,:,~mask_str[0]] = float('nan') 

            # get one-hot versions of the sequence-associated tensors 
            seq_hot, msa, msa_hot, msa_extra_hot, _ = MSAFeaturize_fixbb_inference(seq, params=design_params) 

            # get template featurees 
            t1d = TemplFeaturizeFixbb(seq, conf_1d=conf_1d)
            idx_pdb = torch.from_numpy(np.array(rm.rf)).int()
            t2d   = xyz_to_t2d(xyz_t)
            
            # mask out angles in 'floating points' from t2d input
            if args.floating_points is not None:
                #sanity checks
                if ('-') in args.floating_points:
                    print('WARNING: Cannot make a range of residues floating points!')
                    sys.exit()
                floating_points = [(i[0],int(i[1:])) for i in args.floating_points.split(",")]
                t2d_mask = torch.ones_like(t2d)
                
                for i in floating_points:
                    if i[0].isalpha() is False:
                        print('WARNING: Must specify chain for floating points!')
                        sys.exit()
                    elif (i[0],int(i[1])-1) in mappings['con_ref_pdb_idx'] or (i[0],int(i[1])+1) in mappings['con_ref_pdb_idx']:
                        print('WARNING: floating points must be floating, not joined to other residues!')
                        sys.exit()
                    elif (i[0],int(i[1])) not in mappings['con_ref_pdb_idx']:
                        print('ERROR: floating points must also be specified as residues in the contig string')
                        sys.exit()
                    else:
                        ref_idx = mappings['con_ref_pdb_idx'].index(i)
                        ref_idx0 = mappings['con_hal_idx0'][ref_idx]
                        t2d_mask[:,:,ref_idx0,:,37:] = 0
                        t2d_mask[:,:,:,ref_idx0,37:] = 0          
                t2d = t2d*t2d_mask
           
            
            if args.verbose:
                ic(aa_util.seq2chars(seq))

            ### put on device / cast ###
            seq     = seq.to(DEVICE).long()[None,None].repeat(1,args.n_cycle,1)
            msa     = msa.to(DEVICE).long().unsqueeze(0)
            msa_hot = msa_hot.to(DEVICE).float().unsqueeze(0)
            msa_extra_hot = msa_extra_hot.to(DEVICE).float().unsqueeze(0)
            xyz_t   = xyz_t.to(DEVICE).float().unsqueeze(0)
            t1d = t1d.to(DEVICE).float()[None,None]
            t2d     = t2d.to(DEVICE)
            idx_pdb = idx_pdb.to(DEVICE).unsqueeze(0)
            

            ### run inference ###     
            with torch.no_grad():
                out  = inf_method(model, msa_hot, msa_extra_hot, seq, t1d, t2d, idx_pdb, design_params['MAXCYCLE'])
                logit_c6d, logit_aa, pred_crds, pred_lddts, seq_out = out # unpack 

            lddt = pred_lddts.squeeze().cpu().numpy()
            atoms_out = pred_crds[-1].squeeze()


            strmasktemp = mask_str.squeeze().cpu().numpy()
            partial_lddt = [lddt[i] for i in range(np.shape(strmasktemp)[0]) if strmasktemp[i] == 0]
            trb = {}
            trb['lddt'] = lddt
            trb['inpaint_lddt'] = partial_lddt
            if "-" in args.tmpl_conf:
                trb['tmpl_conf'] = tmpl_conf
            if args.res_translate is not None or args.tie_translate is not None:
                for key, value in translate_dict.items():
                    trb[key] = value
            if args.floating_points is not None:
                trb['floating_points'] = floating_points
            for key, value in mappings.items():
                trb[key] = value
            if args.block_rotate is not None:
                for key,value in rotate_dict.items():

                    if torch.is_tensor(value):
                        value = value.to('cpu').numpy()

                    trb[key] = value

            trb['flags'] = args
            trb['contigs'] = args.contigs

            trb['settings'] = {}
            trb['settings']['pdb'] = args.pdb 

            ### write outputs ###
            if args.dump_npz:
                probs = [torch.nn.functional.softmax(l, dim=1) for l in logit_c6d]
                dict_pred = {}
                dict_pred['p_dist'] = probs[0].permute([0,2,3,1])
                dict_pred['p_omega'] = probs[1].permute([0,2,3,1])
                dict_pred['p_theta'] = probs[2].permute([0,2,3,1])
                dict_pred['p_phi'] = probs[3].permute([0,2,3,1])

                np.savez(f'{args.out}_{i_des}.npz',
                    dist=dict_pred['p_dist'].detach().cpu().numpy()[0,],
                    omega=dict_pred['p_omega'].detach().cpu().numpy()[0,],
                    theta=dict_pred['p_theta'].detach().cpu().numpy()[0,],
                    phi=dict_pred['p_phi'].detach().cpu().numpy()[0,])
            
            atoms_out = pred_crds[-1].squeeze()
            if args.dump_pdb:
                out_prefix = f'{args.out}_{i_des}'
                fname = out_prefix + '.pdb'
                chain_ids = [i[0] for i in rm.hal]
                
                write_pdb(fname, seq_out.squeeze(), atoms_out, Bfacts=lddt, chains=chain_ids)
                '''
                with open(f'{args.out}_{i_des}.out','w') as f:
                    f.write(f'{str(np.mean(lddt))},{str(partial_lddt)}')
                '''
            if args.dump_trb:
                with open(f'{args.out}_{i_des}.trb','wb') as f_out:
                    pickle.dump(trb, f_out)
            '''
            if args.res_translate is not None or args.tie_translate is not None:
                with open(f'{args.out}_{i_des}_translated_coords.json', "w") as outfile:
                    json.dump(translate_dict, outfile)
            '''
    sys.exit('Successfully wrote output')
        

    
if __name__ == '__main__':
    main()
