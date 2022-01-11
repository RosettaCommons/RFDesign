#!/software/conda/envs/SE3/bin/python
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

# absolute path to folder containing this file
script_dir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(script_dir+'/model')
from TrunkModel  import TrunkModule
from model_parsers import parse_a3m, read_templates
from kinematics import xyz_to_t2d
from loss import *
import util 
from scheduler import get_linear_schedule_with_warmup, CosineAnnealingWarmupRestarts
from ffindex import *

# supporting functions
import dj_util, pred_util, prediction_methods 
sys.path.append(script_dir+'/../hallucination/util/') 
import parsers # from ../design/util/

torch.backends.cudnn.deterministic = True
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # debugging, comment out if production 
torch.set_printoptions(sci_mode=False)   # easier for debugging as well 

# suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_args(argv=None):
    """
    Gets command-line arguments (or from a list, if using in Jupyter notebook)
    """
    
    parser = argparse.ArgumentParser()
    
    # i/o args 
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='v00', 
        choices=['v00','v01','v02'], help='Model checkpoint to load') # see main() for paths

    parser.add_argument('--pdb', '-pdb', type=str, default=None,
        help='input pdb file to operate on')                                            

    parser.add_argument('--num_designs', '-N', type=int, default=None,
        help='Number of designs to create')

    parser.add_argument('--start_num', type=int, default=0,
        help='Start number of output designs')

    parser.add_argument('--prefix', type=str, default='',
        help='prefix for dumped files')

    parser.add_argument('--outf', '-outf', type=str, default='./pdbs_test',
        help='directory to spill output into')

    parser.add_argument('--out', '-out', type=str, default=None,
        help='Output path and filename prefix. Overrides --outf and --prefix.')

    parser.add_argument('--input_json', type=str, default=None,
        help='JSON-formatted list of dictionaries, each containing command-line arguments for 1 '\
             'design.')

    parser.add_argument('--dump_pdb', '-dump_pdb', default=False, action='store_true',
        help='dump pdb files into output directories?')

    parser.add_argument('--dump_npz', '-dump_npz', default=False, action='store_true',
        help='dump npz files into output directories?')

    parser.add_argument('--dump_fasta', '-dump_fasta', default=False, action='store_true',
        help='dump fasta files into output directories?')

    parser.add_argument('--dump_all',   '-dump_all',            default=False,          action='store_true',                help='Dump all outputs (pdb, fas, npz, trb)')
    
    # general modelling args 
    parser.add_argument('--task', '-task', type=str, default='des', 
        choices=['des', 'str', 'hal', 'den'],   help='Which prediction mode')

    parser.add_argument('--window', '-w', type=str, default=None,
        help='Regions to mask, e.g. "A,1,10:A,15,15:B,17,24" masks chain A residues 1-10, 15, '\
             '& chain B residues 17-24')

    parser.add_argument('--mask', type=str, default=None,
        help='Regions to mask, alternative syntax to --window, e.g. "A1-10,A15,B17-24" masks chain '\
             'A residues 1-10, 15, & chain B residues 17-24')

    parser.add_argument('--contigs', type=str, default=None,
        help='Which regions to KEEP, with optional gap (masked region) length ranges in '\
             'between, e.g. "A4-10,B70-85" or "5-10,A4-10,6,B70-85,15-20". If gap lengths not '\
             'specified, then --len argument also needed. Used with --task hal')
    
    parser.add_argument('--res_translate',type=str,default=None,
        help='Which residues to translate (randomly in x, y and z direction), '\
             'with maximum distance to translate specified, e.g. "A35,2:B22,4" translates '\
             'residue A35 up to 2A in a random direction, and B22 up to 4A. If specified '\
             'residues are in masked --window, they will be unmasked. In --contig mode, residues'\
             'must not be masked (as need to know where to put them. Default distance to translate is 2A.')
    
    parser.add_argument('--len', type=str, default=None, 
        help='Range of lengths of protein to design. Used with --contigs and --task hal.')

    parser.add_argument('--keep_order', default=False, action='store_true',
        help='When using contig mode and gap regions are not specified, whether to keep '\
             'the contigs (kept regions) in order or shuffle them. If gaps are specified '\
             '(as in A13-25,3-5,A33-41, contigs are always kept in order). Used with --contigs '\
             'and --task hal.')

    parser.add_argument('--min_gap', type=int, default=3,
        help='Mininum length of gap (masked) regions between contigs. Used with --contigs and '\
             '--task hal.')

    parser.add_argument('--mask_seq', type=str, default=None,
        help='Regions to mask (redesign) the sequence only, e.g. "A12,B27-31" will redesign chain '\
             'A residues 12 & chain B residues 27-31')

    parser.add_argument('--receptor_chain', type=str, default=None,
        help='Chain letter for receptor to include in design and output. This chain should be '\
             'present in the input for --pdb. Used with --contigs and --task hal.')

    # native motif as template
    parser.add_argument('--motif_from_trb', type=str, default=None,
        help='Adds template features for native motif, specified by a .trb file generated by '\
             'hallucination. Used when input pdb is a hallucination.')

    parser.add_argument('--motif_from_pdb', type=str, default=None,
        help='Adds template features for native motif, used when input pdb is a hallucination.')

    parser.add_argument('--motif_idx', type=str, default=None,
        help='Index of residues to use as template from motif_from_pdb E.G. "A5,A6,A7,A8-10,A89" '\
             'must be passed with design_idx and be same length ')

    parser.add_argument('--design_idx', type=str, default=None,
        help='Index of residues to apply motif_from_pdb template to E.G. "A1,A2,A3,A4-6,A5" must '\
             'be passed with motif_idx and be same length')

    parser.add_argument('--native_motif_repeat', type=int, default=1,
        help='Number of times to repeat native motif template')

    parser.add_argument('--natmotif_region_weight', type=float, 
        help='Weight (0d/1d template features) for the native motif region in OTHER templates. '\
             'Between 0 and 1, unless -1, which means set 0d/1d features to 0 and also delete '\
             'xyz coordinates.')

    parser.add_argument('--native_motif_only', action='store_true',
        help='Do not keep parent design as fixed template input during recycling.')

    parser.add_argument('--trb_native_ref', action='store_true',
        help='If set, output trb contains mapping between native protein and final design when '\
             'using --motif_from_trb or --motif_from_pdb, as opposed to mapping from seed design '\
             'to final design.' )
    
    # sequence design args 
    parser.add_argument('--inf_method', type=str, default='one_shot',
        choices=['one_shot', 'multi_shot', 'autoregressive', 'msar'],
        help='Which masking/inference  method to use')

    parser.add_argument('--n_iters', type=int, default=2,
        help='If using multi shot, how many iters to use')

    parser.add_argument('--msa_mask_token', type=int, default=21,
        help='Default is 21 for mask token, but can set to 20 (gap) for variation in outputs')

    parser.add_argument('--autoreg_T', type=float, default=1, 
        help='If using autoregressive, sequence sampling temperature')

    parser.add_argument('--init_seq', type=str, default=None, choices=[None, 'poly'],
        help='choose sequence initialization method')

    parser.add_argument('--poly', type=str, default='ALA',
        help='3-letter code of amino acid to initialize unknown sequence with') 

    parser.add_argument('--clamp_1d', type=float, default=None,
        help='Clamp the recycled templated 1D confidences to input float value. Must be in [0,1]')

    parser.add_argument('--local_clamp', default=False, action='store_true',
        help='If True, only clamps 1D features that were in the original mask, rather than '\
             'globally clamping everything')

    parser.add_argument('--motif_template_weight', type=float, default=1,
        help='0d/1d template features values for the native motif. Use with --motif_from_trb.')

    parser.add_argument('--nonmotif_template_weight', type=float, default=1,
        help='0d/1d template features values for the parent design. Use with --motif_from_trb.')

    parser.add_argument('--mask_query', default=False, action='store_true',
        help='Mask entire query sequence (1st row of MSA). Only useful if you have other '\
             'sequences in the MSA.')

    parser.add_argument('--make_ensemble', default=False, action='store_true',
        help='Randomly mutate/mask residues to create an ensemble of structures/sequences')

    parser.add_argument('--make_ensemble_random_walk', default=False, action='store_true',
        help='Iteratively randomly mutate/mask residues to create an ensemble of structures/sequences')

    parser.add_argument('--no_template_accum', default=False, action='store_true',
        help='Do not accumulate templates during random walk.')

    parser.add_argument('--k_ensemble', type=int, default=3,
        help='Number of positions to mask/regenerate during enemble generation')

    parser.add_argument('--indel_rw', default=False, action='store_true',
        help='During ensemble random walk allow to do indels?')

    parser.add_argument('--in_rw', type=int, default=0, help='Number of res to add in indel_rw')

    parser.add_argument('--del_rw', type=int, default=0, help='Number of res to delete in indel_rw')

    parser.add_argument('--recycle_seq', default=False, action='store_true',
        help='Whether to recycle predicted sequence information through iterations')

    parser.add_argument('--recycle_str_mode', type=str, default='pred', choices=['pred','both'],
        help='How to recycle structure info during multi_shot iterations. '\
             '"pred": recycles predicted structure; '\
             '"both": recycles both predicted structure and original template. ')

    # not used yet parser.add_argument('--sample_seq', default=False, action='store_true',
    #    help='When recycling sequence information, will sample from logits if True. Else, argmax.')

    parser.add_argument('--repeat_template', type=int, default=1,
        help='Number of times to repeat template information')

    parser.add_argument('--a3m', type=str, default=None,
        help='path to a3m file to parse and get MSA from')

    parser.add_argument('--hhr', type=str, default=None, help='path to hhr file to parse')

    parser.add_argument('--atab', type=str, default=None, help='HHsearch result file in atab format')

    parser.add_argument('--cautious', default=False, action='store_true',
        help='does not overwrite existing outputs')

    parser.add_argument('--weights_dir', default='/projects/ml/trDesign/autofold/', 
        help='base folder for network weights')

    # check if parsing from list
    if argv is not None:
        args = parser.parse_args(argv) # for use when testing
    else:
        args = parser.parse_args()
  
    # config file override (mainly used to specify alternate --weights_dir)
    if os.path.exists(script_dir + '/config.json'):
        config_opts = json.load(open(script_dir + '/config.json'))
        print('config.json found with the following options (these will override the above):')
        print(config_opts)
        print()
        for k in config_opts:
            setattr(args, k, config_opts[k])

    # add git hash of current commit
    try:
        args.commit = subprocess.check_output(f'git --git-dir {script_dir}/../.git rev-parse HEAD',
                                              shell=True).decode().strip()
    except subprocess.CalledProcessError:
        print('WARNING: Failed to determine git commit hash.')
        args.commit = 'unknown'

    print(f'\nRun settings:\n{args}\n')

    return args
 

def validate_args(args):

    # convert to absolute paths
    if args.pdb is not None: args.pdb = os.path.abspath(args.pdb)
    if args.outf is not None: args.outf = os.path.abspath(args.outf)
    if args.out is not None: args.out = os.path.abspath(args.out)
    if args.motif_from_trb is not None: args.motif_from_trb = os.path.abspath(args.motif_from_trb)
    if args.motif_from_pdb is not None: args.motif_from_pdb = os.path.abspath(args.motif_from_pdb)
    if args.input_json is not None: args.input_json = os.path.abspath(args.input_json)

    # legacy output name specification
    if args.out is None:
        args.out = os.path.join(args.outf, '_'.join((args.prefix, args.task)))

    if args.dump_all:
        args.dump_pdb = True
        args.dump_npz = True
        args.dump_fasta = True

    # sanity checks
    if args.window is None and args.mask is None and args.contigs is None:
        print('WARNING: Neither --window nor --mask nor --contigs were not provided. No masking will happen.')
    if args.window is not None and args.contigs is not None:
        sys.exit('ERROR: Only ONE of --window or --contigs should be provided, not both.')
    if args.pdb is None and args.a3m is None:
        sys.exit('ERROR: if --pdb is not given, --a3m must be given.')
    if args.res_translate is not None and all([x[0].isalpha() for x in args.res_translate.split(':')]) is False:
        sys.exit('ERROR: all residues to translate must have chain specified.')

    return args


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_global_seed(seed):
    """
    Seed all stochasticity
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# temp GRL Q. when length of msa_load is < n_sample, msa_full will be empty?
# parse a3m once and do subsampling for every design run (split this function) ?    
def load_and_sample_msa(a3m):
    """
    Loads an a3m file and creates msa/msa_full by sampling from big one 
    """
    msa_load = torch.from_numpy(parse_a3m(a3m)).long()
    n_sample = 127
    N_seqs   = msa_load.shape[0]

    seq_indices = torch.randperm(N_seqs-1) + 1 # +1 makes sure we never grab msa[0] which is query, will manually add

    latent_selections   = seq_indices[:n_sample]                # indices of latent msa sequences
    full_selections     = seq_indices[n_sample:n_sample+3000]   # indices of full msa sequences

    msa      = msa_load[latent_selections]
    msa      = torch.cat([msa_load[0:1], msa], dim=0)    # manually add query sequence
    msa_full = msa_load[full_selections]
    
    return msa, msa_full

#GRL
def load_top_hhsearch(qlen, db_name, hhr_fn, atab_fn=None, n_templ=10):
    FFindexDB = namedtuple("FFindexDB", "index, data")
    ffdb = FFindexDB(read_index(db_name+'_pdb.ffindex'),
                     read_data(db_name+'_pdb.ffdata'))
    xyz_hhr, t1d, t0d = read_templates(qlen, ffdb, hhr_fn, atab_fn, n_templ=n_templ) #n_templ
    xyz_hhr = xyz_hhr.float().unsqueeze(0)
    t1d = t1d[:,:,2].unsqueeze(-1) # CHECK: took this from data_loader. correct?
    #
    print ('Top hhsearch results loaded as templates')
    ic(xyz_hhr.shape)
    ic(t1d.shape)
    ic(t0d.shape)
    return xyz_hhr, t1d, t0d


def main():
   
    args = get_args()
 
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #GRL. temp. for hhr and atab
    FFDB = "/projects/ml/TrRosetta/pdb100_2020Mar11/pdb100_2020Mar11"
    
    # make model 
    model = TrunkModule(**pred_util.model_param).to(DEVICE)
    model.eval()

    # load checkpoint into model 
    cp_dict = {
        'v00': args.weights_dir+'mixed_masklow.9_maskhigh1.0_fseq2str0.25/BFF_epoch_1_fraction_0.4000091945568224.pt',
        'v01': args.weights_dir+'hal_training_2-20_nan/BFF_epoch_4_fraction_4.597278411180581e-05.pt',
        'v02': args.weights_dir+'jwatson3_first_full_epoch/BFF_epoch_0_fraction_1.0.pt'
    }
    cp_path = cp_dict[args.checkpoint]
    print(f'Loading checkpoint {args.checkpoint}: {cp_path}...')
    ckpt = torch.load(cp_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    print('Successfully loaded pretrained weights into model\n')

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

        # validate arguments
        args = validate_args(args)

        if args.receptor_chain is not None:
            print(f'Receptor being modeled from chain {args.receptor_chain} of input PDB. '\
                  f'Note that only binder sequence is shown in debug output, but receptor is written to output files.')

        # output prefix for this set of arguments
        if args.input_json is None or \
            (args.input_json is not None and \
                ('out' in argdict or 'outf' in argdict or 'prefix' in argdict)):
            # new output name specified in this argdict
            argdict_prefix = args.out
        else:
            # same output prefix, add number to make it unique
            argdict_prefix = args.out+f'_{i_argdict}'

        # get prediction method to use 
        try:
            prediction_method = eval('prediction_methods.'+args.inf_method)
            print('Using inference method',args.inf_method)
        except:
            sys.exit(f'ERROR: Inference method {args.inf_method} not supported.')

        ## Loading up the protein to be operated on
        if args.pdb is not None:
            parsed_pdb      = parsers.parse_pdb(args.pdb)  # can be multi-chain
            xyz_true    = torch.from_numpy(parsed_pdb['xyz'])[:,:3,:]   # N,CA,C
            seq_true    = torch.from_numpy(parsed_pdb['seq'])
        else:
            # no pdb given, spoof one the same length as msa
            msa, msa_full = load_and_sample_msa(args.a3m)
            L_msa = msa.shape[1]
            xyz_true    = torch.full((L_msa,3,3),np.nan)
            seq_true    = msa[0]
            parsed_pdb  = dict(
                seq = seq_true,
                xyz = xyz_true,
                pdb_idx = [('A',i) for i in np.arange(1,L_msa+1)]
            )

        # length of reference protein
        L_ref = seq_true.shape[0]

        # native motif template 
        if args.motif_from_trb is not None:
            # when starting from hallucination
            with open(args.motif_from_trb, 'rb') as f:
                trb_motif = pickle.load(f)
            pdb_native = parsers.parse_pdb(trb_motif['settings']['pdb'])
            xyz_native = torch.tensor(pdb_native['xyz'])[:,:3,:]

        elif args.motif_from_pdb is not None:
            # When starting from pdb
            pdb_native  = parsers.parse_pdb(args.motif_from_pdb)  # can be multi-chain
            xyz_native         = torch.from_numpy(pdb_native['xyz'])[:,:3,:]   # N,CA,C

        # GRL. when we have hhr input. when atab is not given?
        use_hhr = False
        n_templ = 100
        if args.hhr != None and args.atab != None:
            use_hhr = True
            #TODO: n_templ -> need arg. and always take top?
            xyz_hhr, t1d, t0d = load_top_hhsearch(L_ref, FFDB, args.hhr, atab_fn=args.atab, n_templ=n_templ)

        ## loop through designs to make
        prediction_stats_header = ['i_des','seed','RMSD', 'acc1', 'acc2', 'acc3']
        prediction_stats = []
        
        if args.res_translate is not None:
            res_translate = []
            for res in args.res_translate.split(":"):
                try:
                    res_translate.append((res.split(',')[0],float(res.split(',')[1])))
                except:
                    res_translate.append((res.split(',')[0],2.0)) #default distance to 2A if not provided
                    
        for i_des in range(args.start_num, args.num_designs+args.start_num):

            # set seed for reproducibility 
            set_global_seed(i_des)

            # determine output name
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            out_prefix = f'{argdict_prefix}_{i_des}'

            if args.cautious and (os.path.exists(out_prefix+'.npz') or os.path.exists(out_prefix+'.pdb')):
                print(f'\nSkipping design {out_prefix} because output already exists')
                continue

            print(f'\nGenerating design {i_des} ({args.num_designs} total)...')
            t0 = time.time()
            
            if args.res_translate != None:
                xyz_true,translated_coord_dict= dj_util.translate_coords(parsed_pdb, res_translate)
                xyz_true = torch.from_numpy(xyz_true)
                
            # make msa
            seq = torch.clone(seq_true)
            if args.a3m is None:
                msa      = torch.clone(seq_true[None,...]) # add a dimensin so its now [N,L] = [1,L]
                msa_full = torch.clone(msa)
            else:
                # parse the a3m to get MSA 
                msa, msa_full = load_and_sample_msa(args.a3m)
            
            label_aa_s = torch.clone(msa)
                
            # get mask (map kept regions to new protein)
            sm, mappings, mask_str, mask_seq = pred_util.get_mask(args, parsed_pdb)
            L = len(mask_str)

            # # uncomment this to inspect contents of SampledMask object (controls residue mapping)
            #with pd.option_context('display.max_rows', None): 
            #  print(sm.df)
                                      
            # only mask a subsample of the specified mask positions
            # only relevant for window mode
            if args.make_ensemble and (args.window is not None or args.mask is not None):
                k = min(args.k_ensemble, sum(mask_str))
                idx_submask = np.random.choice(np.where(mask_str)[0], k, replace=False)
                mask_str = np.full_like(mask_str, False)
                mask_str[idx_submask] = True # subsampled mask

                # also update SampledMask object (needed for scatter_1d to work correctly)
                new_mask_string = dj_util.SampledMask.contract([pdb_idx for m,pdb_idx in zip(mask_str, sm.ref_pdb_idx) if m])
                sm.inpaint_ranges = [('ref',x) for x in new_mask_string.split(',')]

                # if no special sequence mask specified, make it match structure mask
                if args.mask_seq is None:
                    mask_seq = np.full_like(mask_seq, False)
                    mask_seq[idx_submask] = True

            # contig mode: place kept regions from the reference pdb/msa into new tensors
            if args.contigs is not None:
                xyz_t = sm.scatter_1d(xyz_true, np.nan)[None,None] # (1,1,L,3,3)
                seq = sm.scatter_1d(seq_true, 20) # (L,)
                msa = sm.scatter_1d(msa.permute(1,0), args.msa_mask_token).permute(1,0) # (N,L)
                msa_full = sm.scatter_1d(msa_full.permute(1,0), 
                                         args.msa_mask_token).permute(1,0) # (N,L)
            # window mode and family-based generative mode
            else: 
                # copy input coordinates
                xyz_t = torch.clone(xyz_true)[None,None,:,:3,:] #[1,1,L,3,3] N,Ca,C

            # empty template features
            f0d_t = torch.full((1,1,3), args.nonmotif_template_weight)
            f1d_t = torch.full((1,1,L,1), args.nonmotif_template_weight)

            if args.repeat_template:
                xyz_t = xyz_t.repeat(1,args.repeat_template, 1,1,1)
                f1d_t = f1d_t.repeat(1,args.repeat_template,   1,1)
                f0d_t = f0d_t.repeat(1,args.repeat_template,   1)

            # native motif template features
            # this works for both contig and window modes (for trb input)
            if args.motif_from_trb is not None or args.motif_from_pdb is not None:

                # make SampledMask pointing to native pdb + motif
                sm_motif = sm.copy()

                # get native motif mappings from .trb (from previous halluc / inpainting)
                if args.motif_from_trb is not None:
                    sm_motif.change_ref(
                        old_to_new = dict(zip(trb_motif['con_hal_pdb_idx'], 
                                              trb_motif['con_ref_pdb_idx'])),
                        ref_pdb_idxs_new = pdb_native['pdb_idx'])

                # get native motif mappings from command line arguments
                elif args.motif_from_pdb is not None:
                    sm_motif.change_ref(
                        old_to_new = dict(zip(dj_util.SampledMask.expand(args.design_idx),
                                              dj_util.SampledMask.expand(args.motif_idx))),
                        ref_pdb_idxs_new = pdb_native['pdb_idx'])

                # make native motif templates
                xyz_nat = sm_motif.scatter_1d(xyz_native, np.nan)[None,None] # (1,1,L,3,3)
                xyz_nat = xyz_nat.repeat(1,args.native_motif_repeat,1,1,1)

                con_hal_idx0 = sm_motif.mappings['con_hal_idx0']
                f1d_nat = torch.zeros(1,args.native_motif_repeat,L,1)
                f1d_nat[:,:,con_hal_idx0,:] = args.motif_template_weight

                f0d_nat = torch.full((1,args.native_motif_repeat,3), args.motif_template_weight)

                # downweight native motif region in non-motif templates
                if args.natmotif_region_weight is not None:
                    f1d_t[:,:,con_hal_idx0,:] = max(args.natmotif_region_weight, 0)
                    if args.natmotif_region_weight == -1:
                        xyz_t[:,:,con_hal_idx0] = np.nan

                # append native motif at beginning of templates
                xyz_t = torch.cat([xyz_nat, xyz_t], dim=1)
                f0d_t = torch.cat([f0d_nat, f0d_t], dim=1)
                f1d_t = torch.cat([f1d_nat, f1d_t], dim=1)

            else:
                sm_motif = sm

            # GRL
            # when using hhsearch results as templates
            # subsample within top n_templ and concat under input (xyz_true)
            # TODO: need arg for n_sub_templ
            # TODO: if we don't want to mask? or need different masks?
            # TODO: weights for each template? or confidence?
            if (use_hhr):
                n_sub_templ = min(10, n_templ) #if n_templ<=10, will always subsample the same
                templ_indices = torch.randperm(xyz_hhr.shape[1])
                sub_sel = templ_indices[:n_sub_templ]
                #temp
                print (sub_sel) 
                # concatenate under input true
                xyz_t = torch.cat((xyz_t,xyz_hhr[:,sub_sel,:,:,:]), dim=1)
                f1d_t = torch.cat((f1d_t,t1d[sub_sel,:,:]), dim=0)
                f0d_t = torch.cat((f0d_t,t0d[sub_sel,:]), dim=0)
                #
                if args.pdb is None:
                    # remove spoofed template
                    xyz_t = xyz_t[:,args.repeat_template:]
                    f1d_t = f1d_t[args.repeat_template:]
                    f0d_t = f0d_t[args.repeat_template:]

                print ('Subsampled templates concatenated to input')
                ic(xyz_t.shape)
                ic(f1d_t.shape)
                ic(f0d_t.shape)

            # apply mask to network inputs
            if not args.make_ensemble_random_walk:
                seq, msa, msa_full, xyz_t, f1d_t = \
                    pred_util.mask_inputs(args, mask_str, seq, msa, msa_full, 
                                          xyz_t, f1d_t, mask_seq)            

            t2d   = xyz_to_t2d(xyz_t, f0d_t)    

            #temp GRL
            #print ('After masking')
            #ic(msa.shape)
            #ic(msa_full.shape)
            #ic(xyz_t.shape)
            #ic(f1d_t.shape)
            #ic(f0d_t.shape)
            #ic(t2d.shape)
            
            #make mask_translate to print to output
            if args.res_translate is not None:
                mask_translate = np.zeros(len(mask_str))
                for res in res_translate:
                    try:
                        ref_res = mappings['con_ref_pdb_idx'].index((res[0][0], int(res[0][1:])))
                        hal_res = mappings['con_hal_idx0'][ref_res] #get new position from mapping
                        mask_translate[hal_res] = 1
                    except:
                        print('WARNING: At least one of your --res_translate residues is masked!')
                        
            if args.task != 'des':
                print('mask_str (1=rebuild; 0=keep):')
                print('    '+''.join([str(int(i)) for i,ch in zip(mask_str,sm.hal_pdb_ch) if ch=='A']))
            print('mask_seq (1=rebuild; 0=keep):')
            print('    '+''.join([str(int(i)) for i,ch in zip(mask_seq,sm.hal_pdb_ch) if ch=='A']))
            if args.res_translate is not None:
                print('translated_residues (1=translated; 0=unmoved):')
                print('    '+''.join([str(int(i)) for i in mask_translate]))
            if args.motif_from_trb or args.motif_from_pdb:
                print('native motif (1=motif):')
                print('    '+''.join([str(int(i)) for i,ch in zip(~sm_motif.inpaint, sm.hal_pdb_ch) if ch=='A']))

            # proper batching / device assignment 
            item = {
                'msa'       : msa[None,...].to(DEVICE),
                'msa_full'  : msa_full[None,...].to(DEVICE),
                'seq'       : seq[None,...].to(DEVICE),
                'idx'       : torch.tensor(mappings['idx_rf'])[None].to(DEVICE),
                't0d'       : f0d_t.to(DEVICE),
                't1d'       : f1d_t.to(DEVICE),
                't2d'       : t2d.to(DEVICE),
                'xyz_t'     : xyz_t.to(DEVICE),
                'device'    : DEVICE,
                'args'      : args,
                'mask_str'  : mask_str,
                'mask_seq'  : mask_seq,
                'sm'        : sm,
                'sm_motif'  : sm_motif,
            }

            # make prediction
            with torch.no_grad():
                logit_s, logit_aa_s, logit_tor_s, pred_crds, pred_lddts, pred_seq = \
                    prediction_method(model, item)
                
            #updating L and mask becaus it could change with random walk:
            L=len(pred_seq)
            mask=item['mask_str']
            
            # only compute sequence recovery in window mode
            # it's not defined in contig mode
            if args.window is not None and args.indel_rw is None:

                ## Measure performance at the regions where we wanted a good prediction ##  
                B = 1
                
                # Sequence recovery 
                TOP1,TOP2,TOP3 = 1,2,3
                ic(logit_aa_s.shape)
                logit_aa_s = logit_aa_s.view(B,21,-1,L) # unpack last dim from N*L to N,L
                logits_argsort = torch.argsort(logit_aa_s[:,:,0,:], dim=1, descending=True)
                #ic(logit_aa_s.shape)
                #ic(label_aa_s.shape)
                
                label_aa_s = label_aa_s.to(DEVICE)
                # only compare to 1st sequence of MSA
                acc1 = (label_aa_s[0,mask_seq] == logits_argsort[:, :TOP1, mask_seq.reshape(B,-1).squeeze()]).any(dim=1)
                acc2 = (label_aa_s[0,mask_seq] == logits_argsort[:, :TOP2, mask_seq.reshape(B,-1).squeeze()]).any(dim=1)
                acc3 = (label_aa_s[0,mask_seq] == logits_argsort[:, :TOP3, mask_seq.reshape(B,-1).squeeze()]).any(dim=1)

                ic(acc1.float().mean())
                ic(acc2.float().mean())
                ic(acc3.float().mean())

                # RMSD calculation
                _,rmsd = calc_crd_rmsd(pred_crds, xyz_true[None,...].contiguous().to(DEVICE))
                rmsd = rmsd[-1]

                # append seed / loss information to output list 
                prediction_stats.append([i_des,\
                                         i_des,\
                                         rmsd.detach().cpu().item(),\
                                         acc1.float().mean().cpu().item(),\
                                         acc2.float().mean().cpu().item(),\
                                         acc3.float().mean().cpu().item()])

            ## outputs
            # (chain, res i) numbering used for pdb and fasta outputs
            if args.contigs is not None:
                pdb_idx = mappings['hal_idx1'] # 1-indexed contiguous
            elif args.indel_rw: #indel changes original length so need to account in pdb_idx (L updated earlier)
                L_rec = (sm.hal_pdb_ch=='B').sum()
                pdb_idx = [('A',i) for i in np.arange(1,L+1)] + [('B',i) for i in np.arange(1,L_rec+1)]
            else: # window and non-window prediction modes
                pdb_idx = parsed_pdb['pdb_idx'] # input pdb numbering

            # designed sequence
            # write /'s between chains 
            seq_fasta = ''
            last_chain = pdb_idx[0][0]
            for i,a in zip(pdb_idx, pred_seq):
                if i[0]!=last_chain:
                    seq_fasta += '/'
                    last_chain = i[0]
                seq_fasta += util.aa_N_1[a]

            L_binder = seq_fasta.index('/') if '/' in seq_fasta else L
            print(f'Designed sequence (excluding receptor):\n    {seq_fasta[:L_binder]}')

            print(f'Saving outputs for {out_prefix}: ',end='')

            # Output pdb
            if args.dump_pdb:
                print('pdb ',end='')
                # write to PDB file with the masked regions as BFactors to make
                # sure you got the area you were hoping to mask 
                #ic(pred_crds[-1].squeeze().shape)
                #ic(mask.shape)
                #ic(pred_seq.shape)

                util.writepdb(out_prefix + '.pdb', pred_crds[-1].squeeze(), mask_str, 
                              pred_seq, pdb_idx=pdb_idx)

            # for pyrosetta relaxation / adding sidechains
            if args.dump_npz:
                print('npz ',end='')
                probs = [torch.nn.functional.softmax(l, dim=1) for l in logit_s]
                dict_pred = {}
                dict_pred['p_dist'] = probs[0].permute([0,2,3,1])
                dict_pred['p_omega'] = probs[1].permute([0,2,3,1])
                dict_pred['p_theta'] = probs[2].permute([0,2,3,1])
                dict_pred['p_phi'] = probs[3].permute([0,2,3,1])

                np.savez(out_prefix+'.npz',
                    dist=dict_pred['p_dist'].detach().cpu().numpy()[0,],
                    omega=dict_pred['p_omega'].detach().cpu().numpy()[0,],
                    theta=dict_pred['p_theta'].detach().cpu().numpy()[0,],
                    phi=dict_pred['p_phi'].detach().cpu().numpy()[0,])
           
            # fasta file of sequence
            if args.dump_fasta:
                print('fas ',end='')
                with open(out_prefix+'.fas', 'w') as f_out:
                    print('>' + os.path.basename(out_prefix), file=f_out)
                    print(seq_fasta, file=f_out)

            # design run metadata for downstream metrics like rmsd
            print('trb ',end='')

            if args.res_translate is not None:
                trb = dict(settings = args.__dict__.copy(),
                           out_prefix = out_prefix,
                           mask_str = mask_str,
                           mask_seq = mask_seq,
                           num = i_des,
                           translations = translated_coord_dict
                           )
            else:
                trb = dict(settings = args.__dict__.copy(),
                           out_prefix = out_prefix,
                           mask_str = mask_str,
                           mask_seq = mask_seq,
                           num = i_des
                           )

            # map contig residues from native pdb rather than seed/parent design
            if args.trb_native_ref and (args.motif_from_trb or args.motif_from_pdb):
                trb['settings']['pdb_parent'] = args.pdb
                trb['settings']['pdb'] = trb_motif['settings']['pdb']
                trb['mappings'] = mappings

                idxmap = dict(zip(trb_motif['con_hal_pdb_idx'], trb_motif['con_ref_pdb_idx']))
                sm.change_ref(old_to_new = idxmap, ref_pdb_idxs_new = pdb_native['pdb_idx'])

            trb.update(sm.mappings)

            with open(out_prefix+'.trb', 'wb') as f_out:
                pickle.dump(trb, f_out)

            print()

            print(f'Finished design in {(time.time() - t0):.2f} seconds.')
            print(f'Max CUDA memory: {torch.cuda.max_memory_allocated()/1e9:.4f}G')
            torch.cuda.reset_peak_memory_stats()

        # output stats
        if args.window is not None:
            print('Outputting stats csv')
            pre = '_'.join(out_prefix.split('_')[:-1]) # remove number at end
            with open(pre+'_stats.csv', 'w') as file:
                file.write(','.join(prediction_stats_header) + '\n')
                for line in prediction_stats:
                    file.write(','.join([str(c) for c in line]) + '\n')


if __name__ == '__main__':
    main()
