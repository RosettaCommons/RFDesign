import sys, os, argparse, copy, subprocess, glob, time, pickle, json, tempfile, random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# suppress some pytorch warnings when using a cpu
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, script_dir+'/util')

from models.ensemble import EnsembleNet
from parsers import parse_pdb
from geometry import xyz_to_c6d,c6d_to_bins2

from util import write_pdb, aa_1_N, aa_N_1, alpha_1, combine_pdbs
import util
import kinematics
import contigs
from distutils.util import strtobool

from trFold import TRFold

import loss
import optimization

C6D_KEYS = ['dist','omega','theta','phi']                                                                
torch.backends.cudnn.deterministic = True
torch.set_printoptions(sci_mode=False)   # easier for debugging as well

# only uncomment these 2 lines when debugging! otherwise 2.5x slower
#torch.autograd.set_detect_anomaly(True)
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def get_args(argv=None):

    p = argparse.ArgumentParser()

    # general, I/O
    p.add_argument('--network_name', default='rf_Nov05_2021', help='RosettaFold variant to use')
    p.add_argument('--use_template', type=str, help="'True: Pass the contigs as a template to the network. Contig-like str (ex: A10-15,B3-7): Pass only that subset of the contigs as template.")
    p.add_argument('--num',type=int,default=1,help='number of designs')
    p.add_argument('--start_num',type=int,default=0,help='start at this design number')
    p.add_argument('--msa_num',type=int,default=1,help='number of sequences in MSA')
    p.add_argument('--out',type=str,help='path prefix for output files (e.g. /foo/bar/design)')
    p.add_argument('--cautious',type=strtobool,default=False,help='does not overwrite existing outputs')
    p.add_argument('--save_pdb', type=strtobool, default=True, help='Save a pdb file')
    p.add_argument('--save_npz', type=strtobool, default=True, help='Save a npz file with 6D features')
    p.add_argument('--save_batch_fas', type=strtobool, default=False, help='Save all sequences from a batch')
    p.add_argument('--track_step',type=int,help='save trajectory info every n steps during gradient descent and every 10n steps during MCMC')
    p.add_argument('--track_logits',type=strtobool, default=False, help='Save logits at each "tracked step"')
    p.add_argument('--out_interval',type=int,help='output design every n steps during gradient descent and every 10n steps during MCMC')
    p.add_argument('--out_step',type=str,help='output design on these steps during optimization. e.g. "g20,g30,g100,m25" outputs designs on gradient descent steps 20, 30, and 100, and mcmc step 25')
    p.add_argument('--seed_rng',type=strtobool,default=False,help='seed random number generator with design name to get the same output every time')

    # optimization
    p.add_argument('--steps',type=str, help='comma-separated list of numbers of optimization steps '\
                                            'preceded by a letter indicating the type of optimization. '\
                                            'e.g. "g600,a200" means "do 100 steps of gradient descent '\
                                            'followed by 200 steps of APMC"')
    p.add_argument('--grad_steps',type=int,default=400,help='number of gradient steps to do')
    p.add_argument('--mcmc_steps',type=int,default=0,help='number of mcmc steps to do')
    p.add_argument('--optimizer',type=str,default='nsgd',help='optimizer to use {nsgd, adam}')
    p.add_argument('--drop',type=float,default=0.2,help='dropout to apply to structure prediction module during gradient descent (MCMC will not use dropout)')
    p.add_argument('--init_sd',type=float,default=1e-6,help='stdev of noise to add at logit initialization')
    p.add_argument('--learning_rate',type=float,default=0.05,help='learning rate for gradient descent')
    p.add_argument('--grad_check',type=strtobool,default=True,help='use gradient checkpointing for trunk model')
    p.add_argument('--logit_scale',type=float,default=1,help='initial logits of input sequence when starting gradient descent.')
    p.add_argument('--seq_prob_type', type=str, default='hard', help='<hard, soft, soft2hard> Should the sequences passed to the prediction network be hard (ie one-hot) or soft')
    p.add_argument('--seq_sample', type=strtobool, default=False, help='True: Seq one-hot is sampled from the logit softmax. False: Seq one-hot is logits argmax')
    p.add_argument('--calc_bkg', type=strtobool, default=True, help='True: Averages 100 predictions of random sequences. This is safest, as it takes into account template info and jumps. False: Loads bkg generated with trr_v2. DO NOT USE WHEN PASSING A RECEPTOR.')
    p.add_argument('--cce_sd', type=float, help='Initial sd of logits constrained by cce loss')
    p.add_argument('--hal_sd', type=float, help='Initial sd of logits constrained by free hallucination loss (kl or entropy)')
    
    # constraints
    p.add_argument('--pdb', type=str, required=False, help='path to reference pdb file')
    p.add_argument('--mask', type=str, help='''set gap lengths between contigs of a refernce pdb
    use '<ch>start-end, min_insertion-max_insertion, <ch>start-end, min_insertion-max_insertion ...'
    ex: 'A12-24,2-5,A36-42,20-50,B6-11'
    You can reorder the contigs from the pdb too!
    ex: 'B6-11,9-21,A36-42,20-30,A12-24'
    Gaps and contigs can be in any sequence, though it only makes sense to alternate them
    ex: '10-20,B6-11,9-21,A36-42,20-30,A12-24,22-33'
    ''')    
    p.add_argument('--contigs', type=str, help='Comma separated list of pdb ranges from reference pdb. Ex: A12-19,B39-45,...')
    p.add_argument('--con_set_id', type=str, help='Comma separated list of integers that id each contig belonging to a set')
    p.add_argument('--len', type=str, help='Length range of the hallucinated protein (if using contigs). Ex: 90-110')
    p.add_argument('--keep_order', type=strtobool, default=False, help='Should the contigs be kept in the order provided?')
    p.add_argument('--contig_min_gap', type=int, default=5, help='Minimal length of gap between contigs')
    p.add_argument('--spike', type=float, help='spike the refence protein (--pdb) sequence of the contigs. 0.99=starts native. 0.05=starts random.')
    p.add_argument('--spike_fas', help='Path to fasta file of a single sequence. Use this sequence to spike instead of the pdb sequence.')
    p.add_argument('--force_aa', type=str, help='comma separated list of contig position and aa to force. Ex: A18S (chain A, residue 18 of ref pdb to serine). Listed residues must be a subset of the contig residues.')
    p.add_argument('--exclude_aa', type=str, help='string of AA. Language model will not pick these AA. Only affects sequences from MCMC. Ex: CPG')
    p.add_argument('--force_aa_hal', type=str, help='comma separated list of position and aa to force, in hallucination residue numbers. can include positions outside contigs. only makes sense if mask is specified unambiguously.')
    p.add_argument('--template_pdbs', type=str, nargs='+', help='Space separated list of paths to pdbs to use as templates')
    p.add_argument('--no_bkg_mask', type=strtobool, default=False, 
        help='When True, KL and entropy losses apply to unconstrained AND constrained regions.')
    p.add_argument('--num_repeats', type=int, default=0, help='repeat sequence? (implementation in GD might not be the best)')
    p.add_argument('--init_seq', type=str, default=None, help='string of single amino acid to initialize with Ex: V (only for MCMC currently)')

    #Hallucinate w/ receptor and ligand stub
    p.add_argument('--receptor', type=str, required=False, help='path to reference receptor pdb file')
    p.add_argument('--rec_placement', default='second', help='<"first, second"> Place receptor as the first or second chain.')
    p.add_argument('--gap', type=int, default=200, help='idx between hallucination and receptor')
    
    # loss functions: =1 to turn off, 0 to calculate but not optimize on
    p.add_argument('--w_cce', type=float,    default=1,  help='Weight for cce loss')
    p.add_argument('--include_receptor', type=strtobool, default=False, help='Include receptor in cce calc?')
    p.add_argument('--w_crmsd', type=float,  default=-1, help='Weight for coordinate rmsd loss')
    p.add_argument('--w_entropy', type=float,default=1,  help='Weight for entropy loss')
    p.add_argument('--w_int_entropy', type=float,default=-1,  help='Weight for interface entropy loss')
    p.add_argument('--w_kl', type=float,default=1,  help='Weight for KL divergence loss')
    p.add_argument('--w_rep', type=float,    default=-1, help='Weight for ligand repulsion loss')
    p.add_argument('--w_set_rep', type=float, default=-1, help='Weight for target repulsion loss. Works with contig sets.')
    p.add_argument('--w_atr', type=float,    default=-1, help='Weight for ligand attraction loss')
    p.add_argument('--w_set_atr', type=float, default=-1, help='Weight for target attraction loss. Works with contig sets.')
    p.add_argument('--w_rog', type=float,    default=-1, help='Weight for radius of gyration loss')
    p.add_argument('--w_tgt_atr', type=float,    default=-1, help='Weight for target residue attraction loss')
    p.add_argument('--w_tgt_rep', type=float,    default=-1, help='Weight for target residue repulsion loss')
    p.add_argument('--w_surfnp', type=float,    default=-1,  help='Weight for surface non-polar loss')
    p.add_argument('--w_nc', type=float,    default=-1,  help='Weight for net charge loss')
    p.add_argument('--w_aacomp', type=float,    default=-1,  help='Weight for aa comp loss')
    p.add_argument('--w_sym', type=float,    default=-1, help='Weight for rotational symmetry between sets')
    p.add_argument('--cce_cutoff', type=float, default=19.9, help='Only calculate CCE for residues with Cb < x (angstroms)')
    p.add_argument('--rep_pdb', type=str, default=None, help='PDB file of object on which to apply repulsive loss')
    p.add_argument('--rep_sigma', type=float, default=5, help='Inter-atomic distance for repulsive loss')
    p.add_argument('--atr_pdb', type=str, default=None, help='PDB file of object on which to apply attractive loss.')
    p.add_argument('--atr_sigma', type=float, default=5, help='Inter-atomic distance for attractive loss')
    p.add_argument('--entropy_beta', type=int, default=10, help='Prefactor for modulating distribution sharpness before calculating entropy.')
    p.add_argument('--n_bkg', type=int,      default=100, help='Number of random sequences used to calculate the bkg distribution for KL loss.')
    p.add_argument('--rog_thresh', type=float, default=20, help='radius of gyration below this does not contribute to rog loss')
    p.add_argument('--hot_spot_res', type=str, help='Receptor residues not to mask during site specific free binder hallucination.')
    p.add_argument('--tgt_atr_res', type=str, help='Receptor residues for target residue attraction loss. Used during free binder hallucination.')
    p.add_argument('--tgt_rep_res', type=str, help='Receptor residues for target residue repulsion loss. Used during free binder hallucination.')
    p.add_argument('--surfnp_nbr_thresh', type=float, default=2.5, help='threshold # neighbors to be considered surface residue for surface nonpolar loss')
    p.add_argument('--nc_target', type=float, default=-7, help='target net charge')
    p.add_argument('--entropy_dist_bins', type=int, default=16, help='number of distance bins to use for minimizing entropy')
    p.add_argument('--int_entropy_dist_bins', type=int, default=30, help='number of distance bins to use for minimizing entropy')
    
    # mcmc options
    p.add_argument('--mcmc_halflife', type=float, default=500., help='Halflife (in steps) of temperatures in the MCMC.')
    p.add_argument('--T_acc_0', type=float, default=0.002, help='Initial acceptance temperature during MCMC.')
    p.add_argument('--mcmc_batch', type=int, default=1, help='Batch size for MCMC.')
    p.add_argument('--anneal_t1d', type=strtobool, default=False, help='Reduce template feature confidence from 1 to 0 over mcmc steps. Linear ramp down.')
    p.add_argument('--erode_template', type=strtobool, default=False, help='Gradually reduce the amount of template fed to the network.')
    p.add_argument('--num_masked_tokens', type=int, default=1, help='how many positions to mutate at a time')

    # misc options
    p.add_argument('--weights_dir',type=str,default=script_dir+'/weights/',help='folder containing structre-prediction and language model weights')
    p.add_argument('--device', type=str, default= 'cuda:0', help='Device to run hallucination on cpu or cuda:0; TODO get to work with CPU')
    p.add_argument('--nthreads',type=int,default=4,help='number of cpu threads to use for device="cpu"')
    p.add_argument('--int_topk',type=int,default=-1,help='how many contacts to min on during int_entropy calc. 0 will use all contacts, -1 will use all contacts and ignore no contact bin')
    p.add_argument('--mon_topk',type=int,default=-1,help='how many contacts to min on during monomer entropy calc. 0 will use all contacts, -1 will use all contacts and ignore no contact bin')
    p.add_argument('--cce_cutstep',type=int,default=None,help='cutoff run at step n if CCE isnt below threshold specified with cce_thresh')
    p.add_argument('--cce_thresh',type=float,default=2.2,help='min threshold cce should be below by step specified with cce_cutstep')
    p.add_argument('--entropy_cutstep',type=int,default=None,help='cutoff run at step n if entropy isnt below threshold specified with cce_thresh')
    p.add_argument('--entropy_thresh',type=float,default=2.0,help='min threshold entropy should be below by step specified with cce_cutstep')
    p.add_argument('--int_entropy_cutstep',type=int,default=None,help='cutoff run at step n if entropy isnt below threshold specified with cce_thresh')
    p.add_argument('--int_entropy_thresh',type=float,default=2.0,help='min threshold interface entropy should be below by step specified with cce_cutstep')
    p.add_argument('--rep_cutstep',type=int,default=None,help='cutoff run at step n if rep loss isnt below threshold specified with rep_thresh')
    p.add_argument('--rep_thresh',type=float,default=2.0,help='min rep loss should be below by step specified with rep_cutstep')
    p.add_argument('--n_cycles',type=int, default=1, help='# of cycles to run structure prediction network. only tested on f_Nov05_2021')
    p.add_argument('--init_xyz',type=str,help='initialize coordinates with a guess from this pdb, and use predictions from previous iteration as guess for next iteration. only works for rf_Nov05_2021')
    p.add_argument('--reuse_xyz',type=strtobool, default=False, help='initialize coordinates with a guess from previous iteration. only works for rf_Nov05_2021')
    p.add_argument('--aacomp_dir',type=str,default='./aa_comp.txt',help='folder containing amino acid composition file in residue max_number formate. e.g. "P 2" this means there are max two prolines per design')

    # trFold parameters
    p.add_argument("--trf_batch=", type=int, required=False, dest='batch', default=64, help='trFold batch size')
    p.add_argument("--trf_lr", type=float, required=False, dest='lr', default=0.2, help='trFold learning rate')
    p.add_argument("--trf_nsteps=", type=int, required=False, dest='nsteps', default=100, help='trFold number of minimization steps')    
    
    if argv is not None:
        args = p.parse_args(argv) # for use when testing
    else:
        args = p.parse_args()

    # convert to absolute paths
    args.out = os.path.abspath(args.out)
    if args.pdb is not None: args.pdb = os.path.abspath(args.pdb)

    # sanity checks
    if args.steps is not None:
        for x in args.steps.split(','):
            errstring = 'ERROR: --steps should be comma-separated strings starting with "g" or "m" '\
                        'and ending with an integer. For example: "g400,m2000"'
            if x[0] not in list('gm'):
                sys.exit(errstring)
            try:
                int(x[1:])
            except:
                sys.exit(errstring)
        print('--steps was given. Ignoring --grad_steps, --mcmc_steps.')
    else:
        # convert old syntax ("--grad_steps=100" to new "--steps=g100")
        stages = []
        if args.grad_steps > 0:
            stages.append(f'g{args.grad_steps}')
        if args.mcmc_steps > 0:
            stages.append(f'm{args.mcmc_steps}')
        args.steps = ','.join(stages)
            

    if args.mask is None and args.contigs is None and args.masks_pass is None:
        sys.exit('ERROR: One of --mask, --contigs, or --mask_pass, must be specified.')
        
    if args.pdb is None:
        sys.exit('ERROR: --pdb must be provided.')

    if args.contigs is not None and args.len is None:
        sys.exit('ERROR: When using --contigs, --len must also be given.')

    # add git hash of current commit
    try:
        args.commit = subprocess.check_output(f'git --git-dir {script_dir}/../.git rev-parse HEAD',
                                              shell=True).decode().strip()
    except subprocess.CalledProcessError:
        print('WARNING: Failed to determine git commit hash.')
        args.commit = 'unknown'

    print(f'\nRun settings:\n{args.__dict__}\n')

    return args

def set_global_seed(seed):
    torch.backends.cudnn.deterministic = True

    if type(seed) is str:
        import binascii
        seed = binascii.crc32(seed.encode())

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def canonicalize_force_aa_string(force_aa, pdb):
    '''
    Converts force_aa strings with residue number ranges or without amino acids 
    to a string with one residue number per comma-delimited substring, containing 
    amino acids from the template pdb.
    '''
    idx_map = dict(zip(pdb['pdb_idx'],range(len(pdb['pdb_idx']))))

    force_aa_ = []
    for x in force_aa.split(','):
        if '-' in x or not x[-1].isalpha():
            ch = x[0]
            s,e = contigs.parse_range_string(x[1:])
            force_aa_.extend([ch + str(i) + aa_N_1[pdb['seq'][idx_map[(ch,i)]]] for i in range(s,e+1)])
        else:
            force_aa_.append(x)
    return ','.join(force_aa_)


####################################################
# MAIN
####################################################
def main():

    # arguments & settings
    args = get_args()

    #####################################################
    # Load neural network models
    #####################################################
    device = args.device #'cuda:0'
    torch.set_num_threads(args.nthreads)

    # Structure prediction model: p(structure | sequence)
    Net, net_params = optimization.load_structure_predictor(script_dir, args, device)
    print(f'Loaded sequence-to-structure model {args.network_name} with {sum([p.numel() for p in Net.parameters()])} parameters')
    print(f'\nModel hyperparameters:\n{net_params}')

    # output device, for split-gpu models
    out_device = device
    if hasattr(Net,'c6d_predictor'):
        out_device = next(Net.c6d_predictor.parameters()).device

    # print device names
    devices = list(set([str(device), str(out_device)]))
    print(f'\nUsing CUDA device(s): ',end='')
    for i in devices:
        print(f' {i}: ({torch.cuda.get_device_name(i)}); ',end='')
    print()
  
    #####################################################
    # parse input pdb
    #####################################################
    if args.pdb:
        print('\nParsing input pdb...')
        pdb = parse_pdb(args.pdb)

        if args.receptor:
            print('Parsing receptor pdb...')
            pdb_rec = parse_pdb(args.receptor)
            pdb = combine_pdbs(pdb, pdb_rec, receptor=True)  # adds receptor as chain R
        
        xyz_ref = torch.tensor(pdb['xyz'][:,:,:]).float()
        c6d_ref = xyz_to_c6d(xyz_ref[:,:3,:][None].permute(0,2,1,3),{'DMAX':20.0}).to(device)
        c6d_bins_ref = c6d_to_bins2(c6d_ref,{'DMIN':2.0,'DMAX':20.0,'ABINS':36,'DBINS':36})  # .permute([0,3,1,2])
        pdb['feat'] = c6d_bins_ref.detach().cpu().numpy()
               
    if (args.w_rep >= 0) or (args.w_set_rep >= 0):
        rep_pdb = parse_pdb(args.rep_pdb, parse_hetatom=True)
        rep_xyz = torch.cat([
            torch.tensor(rep_pdb['xyz'][rep_pdb['mask']]),
            torch.tensor(rep_pdb['xyz_het'])
        ]).float()

    if (args.w_atr >= 0) or (args.w_set_atr >= 0):
        atr_pdb = parse_pdb(args.atr_pdb, parse_hetatom=True)
        atr_xyz = torch.cat([
            torch.tensor(atr_pdb['xyz'][atr_pdb['mask']]),
            torch.tensor(atr_pdb['xyz_het'])
        ]).float()
        
    #####################################################
    # Note sets of contigs (each set respects the geometry of other contigs in the same set)
    #####################################################
    if args.con_set_id is not None:
        set_id = [int(l) for l in args.con_set_id.split(',')]
    else:
        set_id = None

    if args.contigs is not None:
        mask = args.contigs
    elif args.mask is not None:
        mask = args.mask
    con_to_set = contigs.mk_con_to_set(mask=mask, 
                                   ref_pdb_idx=pdb['pdb_idx'], 
                                   args=args, 
                                   set_id=set_id
                                  )

    #####################################################
    # Make designs
    #####################################################

    # Loop over the desired number of designs
        
    for n in range(args.start_num, args.start_num + args.num):
        t0 = time.time()

        # determine output filename
        in_prefix= f'{args.out}_{n}'
        out_prefix= f'{args.out}_{n}'

        if args.cautious and os.path.exists(out_prefix+'.npz'):
            print(f'\nSkipping design {out_prefix} because output already exists')
            continue

        #####################################################
        # Sample a specific motif placement
        #####################################################
        # Specific motif segments, place randomly on sequence
        if args.contigs is not None:
            _, mappings = contigs.scatter_contigs(contigs=args.contigs, 
                                                         pdb_out=pdb, 
                                                         L_range=args.len, 
                                                         keep_order=args.keep_order, 
                                                         min_gap=args.contig_min_gap)
        # Specify motif segments and gap lengths
        elif args.mask is not None:
            _, mappings = contigs.apply_mask(args.mask, pdb)
                   
        sampled_mask = mappings['sampled_mask']
        
        # Add receptor contig to sampled mask
        if args.receptor:
          receptor_contig = contigs.get_receptor_contig(pdb['pdb_idx'])
          
          if args.rec_placement == 'first':
            sampled_mask = ','.join([receptor_contig, sampled_mask])
          elif args.rec_placement == 'second':
            sampled_mask = ','.join([sampled_mask, receptor_contig])
        
        # SampledMask object handles motif residue number mapping between
        # reference and hallucination
        sm = contigs.SampledMask(mask_str=sampled_mask,
        ref_pdb_idx=pdb['pdb_idx'], con_to_set=con_to_set)
        
        #####################################################
        # Make 2D masks for losses. All are (L, L)
        #####################################################               
        mask_hall = torch.tensor(sm.get_mask_hal(), 
                                 dtype=float, 
                                 device=device
                                )
        mask_cce = torch.tensor(sm.get_mask_cce(pdb, include_receptor=args.include_receptor), 
                                dtype=float, 
                                device=device
                               )
        
        #####################################################
        # Fill out things for compatibility
        #####################################################
        mappings = {
          'con_ref_idx0': sm.con_mappings['ref_idx0'],
          'con_hal_idx0': sm.con_mappings['hal_idx0'],
          'con_ref_pdb_idx': sm.con_mappings['ref_pdb_idx'],
          'con_hal_pdb_idx': sm.con_mappings['hal_pdb_idx'],
          'sampled_mask': sm.str,
        }

        mask_contig = torch.tensor(sm.get_mask_con())
        
        bins = np.transpose(pdb['feat'], (1,2,3,0))  # (L,L,N,B)
        c6d_bins= sm.scatter_2d(bins)
        c6d_bins= np.transpose(c6d_bins, (3,0,1,2))  #(B,L,L,N)
        c6d_bins= torch.tensor(c6d_bins, dtype=int).to(device)
        
        L = len(sm)

        print(f'\nGenerating {os.path.basename(args.out)}_{n}, length {len(sm)}...')
        # force residues from the receptor
        if args.receptor:
            seq = util.N_to_AA(pdb['seq'])[0]
            ref_pdb_idx = pdb['pdb_idx']

            force_rec_aa = [f'{ch}{resi}{aa}' for aa, (ch, resi) in zip(seq, ref_pdb_idx) 
                            if ch == 'R']
            L_rec = len(force_rec_aa)
            force_rec_aa = ','.join(force_rec_aa)
            
            if args.force_aa:
                args.force_aa = ','.join([args.force_aa, force_rec_aa])
            else:
                args.force_aa = force_rec_aa
        
        if args.receptor:
            L_binder = L - L_rec 
            if args.rec_placement == "first":
                i_binder = np.arange(L_rec, L_rec+L_binder)
                i_receptor = np.arange(L_rec)
            elif args.rec_placement == "second":
                i_binder = np.arange(L_binder)
                i_receptor = np.arange(L_binder, L_binder+L_rec)
        else:
            L_binder = L
            i_binder = np.arange(L_binder)
            i_receptor = []

        # split residue ranges, add aa identities from template pdb if not specified
        if args.force_aa:
            args.force_aa = canonicalize_force_aa_string(args.force_aa, pdb)
            force_aa_display = ','.join([aa for aa in args.force_aa.split(',') if aa[0]!='R'])
            if args.receptor:
                force_aa_display += f', and {L_rec} receptor positions'
            print('Forcing amino acids: ', force_aa_display)
        if args.force_aa_hal:
            print('Forcing amino acids (hallucination residue numbering): '+args.force_aa_hal)

        #####################################################
        # Set up template inputs
        #####################################################
        if 'sm' not in locals():
          sm = None
        net_kwargs = contigs.make_template_features(pdb, args, device, sm_loss=sm)

        if args.init_xyz is not None:
            init_pdb = parse_pdb(args.init_xyz)
            net_kwargs['xyz_prev'] = torch.tensor(init_pdb['xyz'][:,:3][None]).to(device)
            
        #####################################################
        # Define the loss function
        #####################################################

        # precompute functions (computes quantities used by multiple loss functions)
        def superimpose_motif(net_out):
            values = loss.superimpose_pred_xyz(net_out['xyz'], xyz_ref, mappings)
            for k,v in zip(['pred_centroid','ref_centroid','rot'],values):
                net_out[k] = v
        def ligand_distance(net_out):
            xyz_sup = (net_out['xyz'] - net_out['pred_centroid']) @ net_out['rot'][:,None,:,:] \
                      + net_out['ref_centroid'] 
            net_out['lig_dist'] = loss.get_dist_to_ligand(xyz_sup, rep_xyz)
        def n_neighbors(net_out):
            net_out['n_nbrs'] = loss.n_neighbors(net_out['xyz'][:,i_binder])

        # loss functions (wrappers for full functions defined in loss.py)
        def cce_loss(net_out): 
            return loss.get_cce_loss(net_out, mask=torch.tensor(mask_cce).to(out_device),  
                                     c6d_bins=c6d_bins.to(out_device))
        def entropy_loss(net_out): 
            net_out2 = {}
            net_out2['c6d'] = {k:v[:,i_binder][:,:,i_binder] for k,v in net_out['c6d'].items()}
            mask_monomer = mask_hall[i_binder][:,i_binder]
            if args.mon_topk == 0: 
                type_='leaky'
            elif args.mon_topk == -1: 
                type_='orig'
            else:
                type_='leaky_min'
            return loss.get_entropy_loss(net_out2, mask=torch.tensor(mask_monomer).to(out_device), 
                                         beta=args.entropy_beta, type_=type_, k=args.mon_topk, 
                                         dist_bins=args.entropy_dist_bins)
        def crmsd_loss(net_out):
            xyz_sup = (net_out['xyz'] - net_out['pred_centroid']) @ net_out['rot'][:,None,:,:] \
                      + net_out['ref_centroid'] 
            return loss.calc_crd_rmsd(xyz_sup, xyz_ref, mappings)
        def rep_loss(net_out): # Lennard-jones-like repulsion from ligand
            xyz_sup = (net_out['xyz'] - net_out['pred_centroid']) @ net_out['rot'][:,None,:,:] \
                      + net_out['ref_centroid'] 
            lig_dist = loss.get_dist_to_ligand(xyz_sup, rep_xyz)
            return loss.calc_lj_rep(lig_dist, args.rep_sigma)
        def atr_loss(net_out): # Lennard-jones-like attraction to ligand
            xyz_sup = (net_out['xyz'] - net_out['pred_centroid']) @ net_out['rot'][:,None,:,:] \
                      + net_out['ref_centroid'] 
            lig_dist = loss.get_dist_to_ligand(xyz_sup, atr_xyz)
            return loss.calc_lj_atr(lig_dist, args.atr_sigma)
        def rog_loss(net_out):
            return loss.calc_rog(net_out['xyz'][:,i_binder], args.rog_thresh)

        # target residues (for free binder hallucination)
        if args.hot_spot_res is not None:
            res = contigs.SampledMask.expand(args.hot_spot_res)
            res = [('R',i) for ch,i in res] # receptor always parsed as chain R
            idxmap = dict(zip(sm.con_mappings['ref_pdb_idx'],sm.con_mappings['hal_idx0']))
            hot_spot_res = [idxmap[i] for i in res]
            
        if args.tgt_atr_res is not None:
            res = contigs.SampledMask.expand(args.tgt_atr_res)
            res = [('R',i) for ch,i in res] # receptor always parsed as chain R
            idxmap = dict(zip(sm.con_mappings['ref_pdb_idx'],sm.con_mappings['hal_idx0']))
            i_atr_res = [idxmap[i] for i in res]

        if args.tgt_rep_res is not None:
            res = contigs.SampledMask.expand(args.tgt_rep_res)
            res = [('R',i) for ch,i in res] # receptor always parsed as chain R
            idxmap = dict(zip(sm.con_mappings['ref_pdb_idx'],sm.con_mappings['hal_idx0']))
            i_rep_res = [idxmap[i] for i in res]

        def tgt_loss(net_out, i_res, eps = 1e-16):
            bin_centers = torch.linspace(2,20,37) + 0.25
            bin_centers[-1] = 25
            bin_centers = bin_centers.to(net_out['c6d']['p_dist'].device)

            pd = net_out['c6d']['p_dist'][:,:,i_res][:,i_binder] # (B, L_binder, n_target_res, A)
            md = torch.sum(pd*bin_centers, dim=-1) # (B, L_binder, n_target_res)
            md = md.mean(dim=[1,2])
            return md

        def tgt_atr_loss(net_out):
            return tgt_loss(net_out, i_atr_res)
        def tgt_rep_loss(net_out):
            return -tgt_loss(net_out, i_rep_res)

        # interface entropy (for binder hallucination)
        int_mask = torch.ones((L,L)).to(out_device)
        int_mask[i_receptor,:] = 0
        int_mask[:,i_binder] = 0
        if args.hot_spot_res is not None:
            int_mask = torch.zeros((L,L)).to(out_device)
            int_mask[hot_spot_res,:]=1
            int_mask[:,hot_spot_res]=1
            int_mask[i_receptor,:] = 0
            int_mask[:,i_binder] = 0
        
        def int_entropy_loss(net_out):
            if args.int_topk == 0: 
                type_='leaky'
            elif args.int_topk == -1: 
                type_='orig'
            else:
                type_='leaky_min'
            return loss.get_entropy_loss(net_out, mask=int_mask,
                                         beta=args.entropy_beta,
                                         type_=type_,k=args.int_topk,
                                         dist_bins=args.int_entropy_dist_bins)

        # surface nonpolar
        def surfnp_loss(net_out, nonpolar = 'VILMWF', nbr_thresh = args.surfnp_nbr_thresh):
            i_nonpolar = [util.aa_1_N[a] for a in nonpolar]
            surface = 1-torch.sigmoid(net_out['n_nbrs']-nbr_thresh)
            surf_nonpol = net_out['msa_one_hot'][:,0,i_binder][...,i_nonpolar].sum(-1) * surface
            loss = surf_nonpol.sum(-1)/surface.sum(-1)
            return loss

        # net charge
        def nc_loss(net_out, target_charge = args.nc_target):
            i_pos = [util.aa_1_N[a] for a in 'KR']
            i_neg = [util.aa_1_N[a] for a in 'ED']
            charge = net_out['msa_one_hot'][:,0,i_binder][...,i_pos].sum(-1) \
                     - net_out['msa_one_hot'][:,0,i_binder][...,i_neg].sum(-1)
            loss = torch.nn.functional.relu(charge.sum(-1) - target_charge)
            return loss
        
        # aacomp loss 
        aa_comp_dic={}
        if args.w_aacomp>-1:
            lines = []
            with open(args.aacomp_dir) as f:
                lines = f.readlines()
            for line in lines:
                aa_comp_dic[util.aa_1_N[line[0]]]=int(line[2])

        def aacomp_loss (net_out, aacomp = aa_comp_dic):
            loss=torch.tensor(0,device=device)
            for i_pos in aacomp:
                loss = loss + torch.max(net_out['msa_one_hot'][:,0,i_binder][...,i_pos].sum(-1) - aacomp[i_pos],torch.tensor(0,device=device))       
            loss = torch.nn.functional.relu(loss)
            return loss

        def rotation_loss(net_out):
            return loss.calc_rotation_loss(net_out, sm, pdb)
        def set_rep_loss(net_out):
            return loss.calc_set_rep_loss(net_out['xyz'], sm, pdb, rep_pdb, args.rep_sigma)
        def set_atr_loss(net_out):
            return loss.calc_set_atr_loss(net_out['xyz'], sm, pdb, atr_pdb, args.atr_sigma)
        
        # KL loss            
        if args.w_kl >= 0:
            # get bkg distributions
            if args.calc_bkg:
                print(f'Calculating {args.n_bkg} background distributions...')
                bkg = loss.mk_bkg(Net, args.msa_num, L, n_runs=args.n_bkg, net_kwargs=net_kwargs)
            else:
                bkg = np.load(f'/projects/ml/trDesign/backgrounds/generic/{int(L)}.npz')
                bkg = {k: torch.tensor(v).to(out_device) for k, v in bkg.items()}
            def kl_loss(net_out):
                return loss.get_kl_loss(net_out, bkg, mask=torch.tensor(mask_hall).to(out_device))
        else:
            kl_loss = None
                   
        # Construct MultiLoss
        ml = loss.MultiLoss()
        if args.w_crmsd >= 0 or args.w_rep >= 0 or args.w_atr >= 0:
            ml.add_precompute(superimpose_motif)
        if args.w_surfnp >= 0:
            ml.add_precompute(n_neighbors)

        ml.add('cce',        cce_loss,        weight=args.w_cce)
        ml.add('entropy',    entropy_loss,    weight=args.w_entropy)
        ml.add('int_entropy',int_entropy_loss,weight=args.w_int_entropy)
        ml.add('kl',         kl_loss,         weight=args.w_kl)
        ml.add('crmsd',      crmsd_loss,      weight=args.w_crmsd)
        ml.add('rep',        rep_loss,        weight=args.w_rep)
        ml.add('atr',        atr_loss,        weight=args.w_atr)
        ml.add('rog',        rog_loss,        weight=args.w_rog)
        ml.add('tgt_atr',    tgt_atr_loss,    weight=args.w_tgt_atr)
        ml.add('tgt_rep',    tgt_rep_loss,    weight=args.w_tgt_rep)
        ml.add('surfnp',     surfnp_loss,     weight=args.w_surfnp)
        ml.add('nc',         nc_loss,         weight=args.w_nc)
        ml.add('aacomp',     aacomp_loss,     weight=args.w_aacomp)
        ml.add('sym',        rotation_loss,   weight=args.w_sym)
        ml.add('set_rep',    set_rep_loss,    weight=args.w_set_rep)
        ml.add('set_atr',    set_atr_loss,    weight=args.w_set_atr)

        print(ml)
    
        #####################################################
        # Track optimization trajectory metadata
        #####################################################
        trk = {
            'settings': args.__dict__,
            'step': [],
            'step_type': [],
            'loss_tot': [],
        }
        for name in ml.weights.keys():
            trk['loss_'+name] = []

        trb = dict() # best design
        trb['loss_tot'] = 99999
        trb['in_prefix'] = in_prefix
        trb['out_prefix'] = out_prefix
        trb['settings'] = args.__dict__
        trb['net_params'] = net_params
        trb['mask_contig'] = mask_contig.detach().cpu().numpy()
        trb['con_ref_idx0'] = np.array(mappings['con_ref_idx0'])
        trb['con_hal_idx0'] = np.array(mappings['con_hal_idx0'])
        trb['con_ref_pdb_idx'] = mappings['con_ref_pdb_idx']
        trb['con_hal_pdb_idx'] = mappings['con_hal_pdb_idx']
        trb['sampled_mask'] = mappings['sampled_mask']

        if args.receptor is not None:
            rec_pdb_idx = contigs.SampledMask.expand(sm.get_receptor_contig())
            offset = L_binder+1-rec_pdb_idx[0][1]
            if offset > 0: # receptor residue numbers should be greater than binder
                rec_pdb_idx = [(a,b+offset) for a,b in rec_pdb_idx]
            trb['receptor_pdb_idx'] = rec_pdb_idx

        #####################################################
        # Optimization
        #####################################################

        B = 1 #just one batch per design due to GPU memory
        N = args.msa_num

        # Seed RNG
        if args.seed_rng:
            set_global_seed(out_prefix)

        # Initialize the sequence, spiking in ref pdb at contigs
        input_logits = optimization.initialize_logits(args, mappings, L, device, pdb=pdb)

        # parse optimization stages
        stages = [(i, x[0], int(x[1:])) for i,x in enumerate(args.steps.split(','))]

        for i_stage, opt_type, steps in stages:
            #bug here: if g is last step and doesn't give scores lower than best scoring from m; key error for loss print in line 712 (remove comment when fixed)
            print(f'Stage {i_stage}')

            if opt_type == 'g':
                msa, net_out = optimization.gradient_descent(steps, Net, ml, input_logits, 
                                                         args, trb, trk, net_kwargs)
                if msa is None: #if ended before cutoff
                    print("ended trajectory early")
                    break
                    
            elif opt_type == 'm': 
                if 'msa' not in locals(): # when mcmc is the first optimization stage
                    msa = torch.argmax(input_logits, -1).detach().cpu().numpy()
                msa, out = optimization.mcmc(steps, Net, ml, msa, args, trb, trk, net_kwargs, 
                                         sm=sm, pdb=pdb)

        #####################################################
        # Output best design
        #####################################################
        if msa is None:
            # when no design is ouput e.g. when we cut unfruitful trajectories early
            print(f'Aborted design {in_prefix} after {(time.time() - t0)/60:.2f} minutes')
            continue

        if 'msa' not in trb:
            trb['msa'] = msa

        optimization.save_result(out_prefix, Net, ml, trb['msa'], args, trb, 
                                 trk=trk, net_kwargs=net_kwargs)
            
        print(f'Finished design {in_prefix} in {(time.time() - t0)/60:.2f} minutes.')

        if 'msa' in locals(): del msa

if __name__ == "__main__":
    main()
