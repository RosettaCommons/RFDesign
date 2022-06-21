import sys, os, pickle, json, importlib, copy, contigs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, script_dir+'/util')

from parsers import parse_pdb, parse_a3m
from trFold import TRFold
from util import num2aa, write_pdb, aa_1_N, aa_N_1, alphabet_mapping, alpha_1, alphabet_onehot_2_onehot 
import parsers, geometry, util
from models.ensemble import EnsembleNet  

C6D_KEYS = ['dist','omega','theta','phi']

def get_c6d_dict(out, grad=True, args=None):
    C6D_KEYS = ['dist','omega','theta','phi']
    if grad:
        logits = [out[key].float() for key in C6D_KEYS]
    else:
        logits = [out[key].float().detach() for key in C6D_KEYS]
    probs = [F.softmax(l, dim=1) for l in logits]
    dict_pred = {}
    dict_pred['p_dist'] = probs[0].permute([0,2,3,1])
    dict_pred['p_omega'] = probs[1].permute([0,2,3,1])
    dict_pred['p_theta'] = probs[2].permute([0,2,3,1])
    dict_pred['p_phi'] = probs[3].permute([0,2,3,1])
    return dict_pred, probs

def gumbel_st_sample(logits):
    """This function takes logits, constructs softmax and uses Gumbel straight-through to sample the argmax.
    Args:
      logits (torch.float32): [batch_size, num_sequences, sequence_length, 20]
    Returns:
      msa (torch.float32): [batch_size, num_sequences, sequence_length]
    """
    device = logits.device
    B, N, L, _ = logits.shape

    LM_y_logprobs = (logits).log_softmax(-1)
    LM_y_probs = LM_y_logprobs.exp()    
    LM_y_seq = F.one_hot(torch.argmax(LM_y_probs, -1), 21).float() #argmax
    LM_y_seq = (LM_y_seq-LM_y_probs).detach() + LM_y_probs #gumbel trick
    return LM_y_seq

def logits_to_probs(logits, optimizer, step, steps, add_gumbel_noise=False,
                     temp=0.5, eps=1e-8, e_soft = 1, p_soft = 0, e_temp=1e-3, 
                     temp_legacy=1, output_type='soft2hard', learning_rate=0.001):
    
    '''
    Convert sequence logits to a probability distribution, which can be soft,hard, or soft2hard. Can add noise to the process.
    
    Inputs
    ------------
    logits (torch.tensor): Sequence logits. Can be any shape, but typically (B,N,L,A)
    add_gumbel_noise (bool): Roughly equivalent to, "Do you want to sample from the one hot from softmax of the logits?"
    output_type (str, <hard, soft>): Should the ouput be a one-hot or soft distribution?
    temp (float): Temperature of the softmax
    
    Outputs
    ------------
    msa (torch.tensor): Soft, one-hot, or soft2hard probability distribution
    '''
    
    device = logits.device
    B, N, L, A = logits.shape
    
    if add_gumbel_noise:
        U = torch.rand(logits.shape)
        noise = -torch.log(-torch.log(U + eps) + eps)
        noise = noise.to(device)
        logits = logits + noise
    
    y_soft = torch.nn.functional.softmax(logits/temp_legacy, -1)
    if output_type == 'soft':
        msa = y_soft
        if N == 1:
            return msa
    elif output_type == 'hard':
        n_cat = y_soft.shape[-1]
        y_oh = torch.nn.functional.one_hot(y_soft.argmax(-1), n_cat)
        y_hard = (y_oh - y_soft).detach() + y_soft
        msa = y_hard
        if N == 1:
            return msa
    
    # if N > 1, shuffle and block gradients from first sequence
    if N > 1:
        msa = msa[:, torch.randperm(N)]
        msa = torch.cat([msa[:,:1].detach(), msa[:,1:]], axis=1) 
        return msa

    return msa

class NSGD(torch.optim.Optimizer):
  def __init__(self, params, lr, dim):
    defaults = dict(lr=lr)
    super(NSGD, self).__init__(params, defaults)
    self.dim=dim
  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()
    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None: continue
        d_p = p.grad / (torch.norm(p.grad, dim=self.dim, keepdim=True) + 1e-8)
        p.add_(d_p, alpha=-group['lr'])
    return loss
        
def run_trfold(out, probs, args):
    # TRFold
    with torch.enable_grad():
        probs = [p[0] for p in probs]  # Cut batch dimension
        TRF = TRFold(probs)
        if 'xyz' in out.keys(): 
            xyz = TRF.fold(ca_trace=out['xyz'][0,:,1].float(),
                           batch=args.batch,lr=args.lr,nsteps=args.nsteps)
        elif 'lognorm' in out.keys():
            xyz = TRF.fold(lognorm=out['lognorm'].float(),
                           batch=args.batch,lr=args.lr,nsteps=args.nsteps)
        else:
            sys.exit('TRFold cannot generate an initial structure.')
    return xyz
 

def save_result(out_prefix, Net, ml, msa, args, trb, trk=None, net_kwargs={}, loss_inputs={}, msa_one_hot=None ):
    print(f'Saving {out_prefix}: ',end='')

    B,N,L = msa.shape
    trb = copy.copy(trb)
    
    device = next(Net.parameters()).device
    Net.eval()
    if hasattr(Net,'sample'): #to ensure when summing OF models, you don't get average 3D coordinates 
        Net.sample=True
    with torch.no_grad():
        msa_ = torch.tensor(msa).long().to(device)
        if msa_one_hot is not None:
            msa_1h = msa_one_hot
        else:
            msa_1h = F.one_hot(msa_, 21).float()
        out = Net(msa_, msa_one_hot=msa_1h, **net_kwargs)
        dict_pred, probs = get_c6d_dict(out, grad=False)
        net_out = {'c6d': dict_pred, 'xyz': out.get('xyz', None), 'msa_one_hot':msa_1h, 
                   'lddt': out.get('lddt', None),'alpha': out.get('alpha', None)}
        net_out.update(loss_inputs)
        E_0 = ml.score(net_out)
        E_0 = E_0.cpu().numpy()
        
    # sort designs by loss if batch>1. batch 0 is the best design
    if B > 1:
        idx = np.argsort(E_0)
        E_0 = E_0[idx]
        msa = msa[idx]
    
    trb['loss_tot'] = E_0.tolist()
    for name, value in ml.losses.items():
        trb['loss_'+name] = value.cpu().tolist()

    msa_list_NN = msa.reshape(-1,msa.shape[-1]).tolist()
    msa_list_AA = alphabet_mapping(msa_list_NN, aa_N_1)
    seqs_fasta = ["".join(msa_list_AA[i]) for i in range(B)]

    trb['seq'] = seqs_fasta[0]

    dict_pred_out = {
        k: dict_pred[k].detach().cpu().numpy()
        for k in ['p_dist','p_omega','p_theta','p_phi']
    }

    if args.receptor:
      # always make the receptor chain B in the fasta
      idx_tmpl = net_kwargs['idx'].cpu().numpy()[0]
      ch_break_idx = np.where(np.diff(idx_tmpl) > 1)[0][0] + 1
      
      seqs_fasta_ = []
      for seq in seqs_fasta:
        seq_ch0 = seq[:ch_break_idx]
        seq_ch1 = seq[ch_break_idx:]
        if args.rec_placement == 'first':
          seqs_fasta_.append(f'{seq_ch1}/{seq_ch0}')
        elif args.rec_placement == 'second':
          seqs_fasta_.append(f'{seq_ch0}/{seq_ch1}')
      seqs_fasta = seqs_fasta_
          
      # make the receptor "chain B" in the output c6d
      if args.rec_placement == 'first':
        map_chB_last = np.concatenate([np.arange(ch_break_idx, L), np.arange(0, ch_break_idx)])
      elif args.rec_placement == 'second':
        map_chB_last = np.arange(L)
        
      for k,v in dict_pred_out.items():
          dict_pred_out[k] = v[:,map_chB_last[:,None], map_chB_last[None,:]]
 
    if args.save_npz:
        print('npz',end='')
        np.savez(out_prefix+'.npz',
            dist=dict_pred_out['p_dist'][0],
            omega=dict_pred_out['p_omega'][0],
            theta=dict_pred_out['p_theta'][0],
            phi=dict_pred_out['p_phi'][0],
            mask=trb['mask_contig'])
   
    # save sequence from best batch
    print(', fas',end='')
    with open(out_prefix+'.fas', "w") as f_out:
        print(">" + os.path.basename(out_prefix), file=f_out)
        print(seqs_fasta[0], file=f_out)
            
    # save sequence from all batches
    if 'save_batch_fas' in args and args.save_batch_fas:
        print(', batch.fas',end='')
        with open(out_prefix+'_batch.fas', "w") as f_out:
            for i in range(B):
                print(">" + os.path.basename(out_prefix+f"_b{i}"), file=f_out)
                print(seqs_fasta[i], file=f_out)

    # Save tracker
    with open(out_prefix+'.trb', 'wb') as f_out:
        print(', trb',end='')
        pickle.dump(trb, f_out)

    if trk is not None:
        print(', trk',end='')
        with open(out_prefix+'.trk', 'wb') as f_out:
            pickle.dump(trk, f_out)
        
    # Save pdb
    if args.save_pdb and 'xyz' in out:
        print(', trfold pdb',end='')
        xyz = run_trfold(out, probs, args).permute(1,0,2)
        Bfacts = None
        sc = False
               
        # write file
        comments = ['##### Meta data #####']
        comments += [f'{k}: {v}' for k, v in args.__dict__.items()]
        comments += [f'{k}: {trb[k]}' for k in ['con_ref_idx0',
                                                'con_hal_idx0',
                                                'con_ref_pdb_idx',
                                                'con_hal_pdb_idx',
                                                'sampled_mask']]

        if '/' in seqs_fasta[0]:
            chain_break = seqs_fasta[0].index('/')
            pdb_idx = [('A',i) for i in range(1,chain_break+1)] + \
                      [('B', idx[1]) for idx in trb['receptor_pdb_idx']]
        else:
            pdb_idx = [('A',i) for i in range(1,L+1)]

        write_pdb(xyz=xyz.detach().cpu().numpy(), #.permute(1,0,2)
                  prefix=out_prefix,
                  res=seqs_fasta[0].replace('/',''),
                  pdb_idx=pdb_idx,
                  Bfacts = Bfacts,
                  sc = sc,
                  comments=comments)             

    print() # newline
    

def load_structure_predictor(include_dir, args, device):
    def import_method(path_str):
        '''import a method from folder/subfolder/file.py given a string
        folder.subfolder.file.method; add folder/subfolder to sys.path'''
        module,method = path_str.rsplit('.',1)
        sys.path.insert(0, include_dir+'/'+module.rsplit('.',1)[0].replace('.','/'))
        module = importlib.import_module(module)
        method = getattr(module,method)
        return method

    print(f'Loading structure prediction model onto device {device}...')
    reg_models = json.load(open(include_dir+"/models/models.json"))
    
    for k,v in reg_models.items():
        flag = '*' if k==args.network_name else ' '
        if k==args.network_name or v['visible']:
            print("# %c %-16s  [ens=%d]   %s"%(flag,k,len(v['checkpoints']),v['name']))

    sel = reg_models[args.network_name]
    chks = [args.weights_dir+'/'+sel['weights_path']+'/'+chk for chk in sel['checkpoints']]
    NetClass = import_method(sel['code_path'])

    net_params = json.load(open(args.weights_dir+'/'+sel['weights_path']+'/'+sel['params_path']))
    Net = NetClass(**net_params)
    weights = torch.load(chks[0],map_location=torch.device(device))
    if 'model_state_dict' in weights.keys():
        weights = weights['model_state_dict']
    Net.load_state_dict(weights, strict=False)

    Net = Net.to(device)
    Net.eval()
    for p in Net.parameters(): p.requires_grad = False        
    
    return Net, net_params

def uniform_noise(seq, p):
    '''
    Converts a sequence string to logits, with non-sequence aa logits set to some constant.
    
    Inputs
    --------------
    seq (str): String of AA1
    p (float): Frequency at which input aa will be sampled (using softmax B=1)
    
    Outputs
    --------------
    input_logits (torch.tensor, (B,N,L,21)): Logits to start hallucination from
    
    '''
    # Checking input type
    assert isinstance(seq, str), 'seq must be a str'
    assert isinstance(p, float), 'p must be a float'
    
    # convert to one hot
    n_cat = torch.tensor(20)
    aan = torch.tensor([util.aa_1_N[s] for s in seq], dtype=torch.long)
    seq_oh = torch.eye(n_cat)[aan]

    # scale st seed aa will be sampled at desired frequency p
    p = torch.tensor(p).float()
    g = torch.log(p) - torch.log(1 - p) + torch.log(n_cat.float() - 1.)
    input_logits = seq_oh * g
    
    # make gap character very unlikely
    L = input_logits.shape[0]
    gap_logits = -1e4 * torch.ones([L,1])
    input_logits = torch.cat([input_logits, gap_logits], -1)
    
    # add batch (B) and n_seq (N) dimensions
    input_logits = input_logits[None, None, :, :]
    
    return input_logits

def add_gumbel_noise(seq, p, eps=1e-8):
    '''
    Converts seq to logits such that the wt sequence is the argmax p fraction
    of the time. Just a fancy way of saying this function randomly mutates 
    p fraction of the input sequence and returns the logit representation.

    Inputs
    --------------
    seq (str): String of AA1
    p (float): Frequency at which input aa will be sampled (using softmax B=1)

    Outputs
    --------------
    input_logits (torch.tensor, (B,N,L,21)): Logits to start hallucination from

    '''
    logits = uniform_noise(seq, p)

    # add gumbel noise
    U = torch.rand(logits.shape)
    noise = -torch.log(-torch.log(U + eps) + eps)
    logits_noised = logits + noise

    return logits_noised


def initialize_logits(args, mappings, L, device, B=1, pdb=None,init_zeroes=False):
    '''
    Note: These options are not necessarily mutually exclusive!
    '''
    N = args.msa_num
   
    if (args.cce_sd is not None) and (args.hal_sd is not None):
        # Initialize logits for the cce and free hallucination regions with different init_sd
        m_cce = mappings['mask_1d'][0].astype(bool)
        m_cce = torch.tensor(m_cce)
        m_hal = ~m_cce
        
        n_cce = m_cce.sum()
        n_hal = m_hal.sum()
        
        input_logits = torch.zeros(B, args.msa_num, L, 21)
        input_logits[:, :, m_cce] = args.cce_sd * torch.randn(B, args.msa_num, n_cce, 21)
        input_logits[:, :, m_hal] = args.hal_sd * torch.randn(B, args.msa_num, n_hal, 21)
        
        # Make gaps very unlikely
        input_logits[...,20] = -1e9
                   
    elif init_zeroes==True:
        print("init zeroes")
        input_logits=torch.zeros(B,N,L,21).to(device).float()
        input_logits[...,20] = -1e9 # avoid gaps
    else:
        # Random initialization
        input_logits = args.init_sd*torch.randn([B,N,L,21]).to(device).float()
        input_logits[...,20] = -1e9 # avoid gaps
        
    if args.spike is not None or args.spike_fas is not None:
        # Spike in sequence from the contigs or supplied fasta file
        print('Initializing logits: Spike')

        # input fasta sequence
        if args.spike_fas is not None:
            fas = parse_a3m(args.spike_fas)
            seq_ref = torch.tensor(fas['msa'][0]).long()

            # only go up to "/" character (remove receptor)
            idx = torch.where(seq_ref > 19)[0]
            if len(idx) > 0:
                seq_ref = seq_ref[:idx[0]] 

            # add receptor sequence from input pdb  
            if args.receptor is not None:
                pdb_rec = parse_pdb(args.receptor)
                if args.rec_placement == 'second':
                    seq_ref = torch.cat([seq_ref, torch.tensor(pdb_rec['seq'])])
                else:
                    seq_ref = torch.cat([torch.tensor(pdb_rec['seq']), seq_ref])

            seq_init = seq_ref[None,None]

        # scatter ref contig sequence to hal contigs
        else:
            seq_init = torch.zeros([B,1,L], dtype=torch.int64)
            seq_ref = torch.tensor(pdb['seq'], dtype=torch.int64)
            seq_init[:,:, mappings['con_hal_idx0']] = seq_ref[mappings['con_ref_idx0']]

        n_cat = torch.tensor(20)
        seq_init = F.one_hot(seq_init, n_cat).float()  # (B,1,L,20)

        #2. weight logits so they are sampled at desired frequency
        p = torch.tensor(args.spike, dtype=torch.float32)
        g = torch.log(p) - torch.log(1 - p) + torch.log(n_cat.float() - 1.)  # desired gap in logit magnitude
        seq_init *= g
        
        #4. Extend to an MSA  (spiking only really makes sense if 1 seqeunce in the MSA)
        update = torch.tensor( #[N,L,20]
            torch.cat([torch.cat([seq_init, 
                                  -1e6*torch.ones([B,1,L,1])],  # make gap very unlikely
                                 axis=-1),
                       torch.zeros([B,N-1,L,21])],  # rest of msa starts at 0
                      axis=1) 
            + args.init_sd*torch.randn([B,N,L,21]),  # add a little noise
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )
        
        #4. Update input_logits
        input_logits = input_logits + update
        
    # copy and turn on gradients
    input_logits = torch.tensor(input_logits).to(device).requires_grad_(True)

    return input_logits


def enable_dropout(Net):
    # turn on dropout
    for m in Net.modules():
      if m.__class__.__name__.startswith('Dropout'):
        m.train()
    return Net
     
def force_msa(msa, alphabet, force_aa=None, exclude_aa=None, weight=1e8):
    '''
    Force and exclude certain amino acids in an MSA
    
    Inputs
    -------------
    msa: (torch.tensor, (B,N,L,A))
        Logits of a multiple sequence alignment
    alphabet: (dict)
        Mapping one-letter amino acids (upper case) to int
    exclude_aa: (List)
        List of AA1 to exclude.
    force_aa: (dict)
        key: (int) idx0 of the hallucinated protein.
        val: (str) One letter amino acid
        Forces certain amino acids at certain positions.
        Supersedes exclude_aa
    weight: (float)
        How much to increase/decrease the logit
        
    Outputs
    -------------
    msa: (torch.tensor, (B,N,L,A))
        Updated msa.
    '''
    B,N,L,A = msa.shape
    dtype = msa.dtype
    device = msa.device
    bias = torch.zeros((1,1,L,A), dtype=dtype, device=device)
    
    if exclude_aa is not None:
        for aa in exclude_aa:
            aa = aa.upper()
            bias[0, 0, :, alphabet[aa]] = -1
        
    if force_aa is not None:        
        for idx0, aa in force_aa.items():
            aa = aa.upper()
            bias[0, 0, idx0, alphabet[aa]] = 1
            
    msa = msa + weight * bias
    
    return msa

def stop_early(args,trb,step):
    if args.cce_cutstep is not None and step == args.cce_cutstep:
        if trb['loss_cce'] > args.cce_thresh:
            print(f"CCE loss: {trb['loss_cce']} of best total loss below cutoff of {args.cce_thresh} at step {step} so terminating early")
            return True
    if args.rep_cutstep is not None and step == args.rep_cutstep:
        if trb['loss_rep'] > args.rep_thresh:
            print(f"rep loss: {trb['loss_rep']} of best total loss below cutoff of {args.rep_thresh} at step {step} so terminating early")
            return True
    if args.entropy_cutstep is not None and step == args.entropy_cutstep:
        if trb['loss_entropy'] > args.entropy_thresh:
            print(f"entropy loss: {trb['loss_entropy']} of best total loss below cutoff of {args.entropy_thresh} at step {step} so terminating early")
            return True
    if args.int_entropy_cutstep is not None and step == args.int_entropy_cutstep:
        if trb['loss_int_entropy'] > args.int_entropy_thresh:
            print(f"int_entropy loss: {trb['loss_int_entropy']} of best total loss below cutoff of {args.int_entropy_thresh} at step {step} so terminating early")
            return True
    
       

def gradient_descent(steps, Net, ml, input_logits, args, trb, trk, net_kwargs={}):

    print('Starting gradient descent...')
    print(f'{"step":>12}',end='')
    ml.print_header_line()

    device = next(Net.parameters()).device

    torch.set_grad_enabled(True)
    if args.drop > 0: Net = enable_dropout(Net)

    # gradient checkpointing
    if args.grad_check and ('trunk' in args.network_name or 'rf' in args.network_name):
        net_kwargs = copy.copy(net_kwargs)
        net_kwargs['use_transf_checkpoint'] = True

    B,N,L,A = input_logits.shape

    # turn off gradients to make in-place modifications
    input_logits.requires_grad_(False)

    # exclude these AAs at all positions
    exclude_aa = [aa_1_N['-']] # always exclude gaps
    if args.exclude_aa is not None: 
        exclude_aa += [aa_1_N[aa.upper()] for aa in list(args.exclude_aa)]

    # keep these amino acids fixed at these positions
    if args.force_aa is not None:
        idxmap = dict(zip(trb['con_ref_pdb_idx'], trb['con_hal_idx0']))
        idx_force = [idxmap[(x[0], int(x[1:-1]))] for x in args.force_aa.split(',')]
        force_aa = [aa_1_N[x[-1].upper()] for x in args.force_aa.split(',')]
    else:
        idx_force, force_aa = [], []
   
    if args.force_aa_hal is not None:
        idx0_hal, aa_s = [], [] 
        idx0_hal = idx0_hal + [int(x[1:-1])-1 for x in args.force_aa_hal.split(',')]
        aa_s = aa_s + [aa_1_N[x[-1].upper()] for x in args.force_aa_hal.split(',')]
        idx_force.extend(idx0_hal)
        force_aa.extend(aa_s)

    # prepare optimizer
    input_logits = torch.tensor(input_logits).to(device).requires_grad_(True)
    if args.optimizer == 'nsgd':
        optimizer = NSGD([input_logits], lr=args.learning_rate*np.sqrt(L), dim=[-1,-2])
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam([input_logits], lr=args.learning_rate)

    msa = torch.argmax(input_logits, -1).detach().cpu().numpy()
    best_step=0

    for step in range(steps+1): # 1 extra iteration to compute final outputs
        optimizer.zero_grad()
        
        if stop_early(args,trb,step):
            return None, None


        # no update on last iteration
        if step == steps: 
            torch.set_grad_enabled(False)

        # force / exclude AAs
        bias = torch.zeros_like(input_logits)
        for aa in exclude_aa:
            bias[0, 0, :, aa] = bias[0, 0, :, aa] - 1e9
        for i, aa in zip(idx_force, force_aa):
            bias[0, 0, i, aa] = bias[0, 0, i, aa] + 1e18
        input_logits_biased = input_logits + bias

        #Force repeat sequence w/ logits 
        if args.num_repeats > 1:
            #other way to do this: find aa most prefered by all repeat positions, add bias to that position s.t. its the most probable
            L_repeat=math.floor(input_logits.shape[2]/args.num_repeats)
            for r in range(1,args.num_repeats):
                start=L_repeat*r
                end=start+L_repeat
                input_logits_biased[:,:,start:end,:]=input_logits_biased[:,:,0:L_repeat,:]

        
        # gumbel-softmax sampling
        msa_one_hot = logits_to_probs(input_logits_biased, optimizer, step, steps+1, add_gumbel_noise=args.seq_sample,
                                        learning_rate=args.learning_rate ,output_type=args.seq_prob_type)  
        
        out = Net(torch.argmax(msa_one_hot,axis=-1).to(device), msa_one_hot=msa_one_hot.to(device),
                  **net_kwargs)
        dict_pred, probs = get_c6d_dict(out,args=args) 

        # calculate loss
        net_out = {'c6d': dict_pred, 'xyz': out.get('xyz', None), 'msa_one_hot': msa_one_hot, 'lddt': out.get('lddt', None),'alpha': out.get('alpha', None)} 
        E_0 = ml.score(net_out)
         
        # initial coordinates for next iteration
        if args.reuse_xyz:
            net_kwargs['xyz_prev'] = out['xyz'].detach()

        # track intermediate losses
        if args.track_step is not None and step % args.track_step == 0:
            trk['step'].append(int(step))
            trk['step_type'].append('grad')
            trk['loss_tot'].append(float(E_0))
            for name, value in ml.losses.items():
                trk['loss_'+name].append(float(value))
            if args.track_logits:
                if 'input_logits' in trk:
                    trk['input_logits'].append(input_logits.clone())
                    trk['msa_one_hot'].append(msa_one_hot.clone())
                else:
                    trk['input_logits'] = [input_logits.clone()]
                    trk['msa_one_hot'] = [msa_one_hot.clone()]
                    

        # best design so far
        if E_0 < trb['loss_tot']:
            trb['loss_tot'] = float(E_0)
            for name, value in ml.losses.items():
                trb['loss_'+name] = float(value)
            msa = torch.argmax(msa_one_hot.detach(),axis=-1).cpu().numpy()  #(B,N,L), B=1, N=1
            trb['msa'] = msa
            best_step=step
       
        if step % 10 == 0:
            print(f'{step:>12}', end='')
            ml.print_losses_line()

        # output intermediate result
        out_step = []
        if args.out_interval is not None:
            out_step.extend(np.arange(args.out_interval,steps,args.out_interval))
        if args.out_step is not None:
            out_step.extend([int(x[1:]) for x in args.out_step.split(',') if x[0]=='g'])
        if step in out_step:
            save_result(f'{trb["out_prefix"]}_g{step}', Net, ml, msa, args, trb, trk, 
                        net_kwargs=net_kwargs,msa_one_hot=msa_one_hot)
            if args.drop > 0: Net = enable_dropout(Net) # dropout gets turned off when saving result

        if step != steps: # no update on last iteration
            E_0.backward()
            optimizer.step()

    # final loss
    print(f'{"final":>12}',end='')
    print(f'{trb["loss_tot"]:>12.4f}',end='')
    for name, value in ml.losses.items():
        print(f'{trb["loss_"+name]:>12.4f}',end='')
    print()
    print("best loss step:",best_step)
    print(f'Max CUDA memory: {torch.cuda.max_memory_allocated()/1e9:.4f}G')
    torch.cuda.reset_peak_memory_stats()
    
    return msa, out
    
def uniform_sampler(sequences, num_masked_tokens, do_not_mutate=[], exclude_aa=None,args=None):
    """
    Args:
    sequences: [list] a list of sequences; sequences = ['AGRGA', 'AGAGA', 'AGPGA']
    num_masked_tokens: [int32], number of AAs to mask
    T_1: [float32], float for temperature of LM; high temp -> uniform distribution
    do_not_mutate: [list of ints] index1 positions the LM model will not mutate
    Returns:
    mutated_seq_list: [list] a list of sequences; sequences = ['AGRGA', 'APPPA', 'AGPGA']
    """
    if exclude_aa is None:
        exclude_aa = ''
    mutated = []
    alpha = list("ARNDCQEGHILKMFPSTWYV-")
    alpha = [a for a in alpha if a not in exclude_aa and a != '-']

    for seq in sequences:
        do_not_mutate = np.array(do_not_mutate, dtype=int) - 1
        can_mutate_bool = np.ones(len(seq), dtype=bool)
        can_mutate_bool[do_not_mutate] = False
        if args.num_repeats>1:
            #only mutate first instance of repeat
            L_repeat=math.floor(len(seq)/args.num_repeats)
            can_mutate_bool[L_repeat:]= False
        can_mutate_idx = np.arange(len(seq), dtype=int)[can_mutate_bool]
        
        idx = np.random.choice(can_mutate_idx, num_masked_tokens, replace=False)
        for i in idx:
            aa_mut = alpha[np.random.randint(0,len(alpha))]
            seq[i] = aa_mut
            if args.num_repeats>1:
                #apply mutation to all repeat units
                for r in range(0,args.num_repeats):
                    if r != 0:
                            i=i+L_repeat
                    seq[i] = aa_mut
        mutated.append(''.join(seq))
    return mutated

def mcmc(steps, Net, ml, msa, args, trb, trk, net_kwargs={}, 
         sampler=uniform_sampler, sm=None, pdb=None):
    '''
    Inputs
    ---------
    sm: Instance of SampledMask. Keeps track of where contigs and receptors are
    '''

    Net.eval() # turn off dropout for MCMC
    torch.set_grad_enabled(False)

    device = next(Net.parameters()).device
    out_device = device                                                                            
    if hasattr(Net,'c6d_predictor'):
        out_device = next(Net.c6d_predictor.parameters()).device

    # Allow batches larger than 1
    B, N, L = msa.shape    
    if (B == 1) and (N == 1):
        msa = np.tile(msa, (args.mcmc_batch,1,1))
        B, N, L = msa.shape

    # Template features
    if args.mcmc_batch > 1 and args.use_template:
        net_kwargs['idx'] = net_kwargs['idx'].repeat(B,1)
        net_kwargs['t1d'] = net_kwargs['t1d'].repeat(B,1,1,1)
        net_kwargs['t2d'] = net_kwargs['t2d'].repeat(B,1,1,1,1)

    if args.init_seq is not None:
        #force all positions ot this aa:
        msa[0, 0,:]=aa_1_N[args.init_seq.upper()]
    
    if args.num_repeats>1:
        #make sure starting seqeunce is a random repeat
        L_repeat=math.floor(msa.shape[2]/args.num_repeats)
        for r in range(1,args.num_repeats):
            start=L_repeat*r
            end=start+L_repeat
            msa[:,:,start:end]=msa[:,:,0:L_repeat]

    # Prevent LM sampler from mutating forced positions
    do_not_mutate, idx0_hal, aa_s = [], [], []
    if args.force_aa is not None:
        idxmap = dict(zip(trb['con_ref_pdb_idx'], trb['con_hal_idx0']))
        idx0_hal = idx0_hal + [idxmap[(x[0], int(x[1:-1]))] for x in args.force_aa.split(',')]
        aa_s = aa_s + [x[-1] for x in args.force_aa.split(',')]

    if args.force_aa_hal is not None:
        idx0_hal = idx0_hal + [int(x[1:-1])-1 for x in args.force_aa_hal.split(',')]
        aa_s = aa_s + [x[-1] for x in args.force_aa_hal.split(',')]
        
    # set desired AAs in input msa
    for i, aa in zip(idx0_hal, aa_s):
        msa[0, 0, i] = aa_1_N[aa.upper()]

    # prevent these positions from being mutated
    do_not_mutate = [i+1 for i in idx0_hal]

    print('Starting MCMC...')
    print(f'{"step":>12}',end='')
    ml.print_header_line()

    sf_0 = np.exp(np.log(0.5) / args.mcmc_halflife)  # scaling factor / step
    
    # initial score
    msa_ = torch.tensor(msa).long().to(device)
    msa_1h = F.one_hot(msa_,21).float()

    # hack for nvidia SE3 RF models: 1st forward pass gives error, but later ones work
    for i in range(2):
        try:
            out = Net(msa_, msa_one_hot=msa_1h, **net_kwargs)
        except:
            pass
 
    out = Net(msa_, msa_one_hot=msa_1h, **net_kwargs)
    dict_pred, probs = get_c6d_dict(out, grad=False)
    net_out = {'c6d': dict_pred, 'xyz': out.get('xyz', None), 'msa_one_hot':msa_1h, 'lddt': out.get('lddt', None),'alpha': out.get('alpha', None)}
    E_0 = ml.score(net_out)

    for step in range(steps):
        sf = sf_0 ** step
        msa_list_NN = msa.reshape(-1,msa.shape[-1]).tolist()
        msa_list_AA = alphabet_mapping(msa_list_NN, aa_N_1)
        msa_list_AA_mutated = sampler(msa_list_AA, args.num_masked_tokens, do_not_mutate=do_not_mutate, 
                                      exclude_aa=args.exclude_aa,args=args)
        msa_list_NN_mutated = alphabet_mapping(msa_list_AA_mutated, aa_1_N)
        mutated_msa = np.array(msa_list_NN_mutated).reshape(B, N, L)
       
        msa_ = torch.tensor(mutated_msa).long().to(device)
        msa_1h = F.one_hot(msa_,21).float()

        out = Net(msa_, msa_one_hot=msa_1h, **net_kwargs)

        dict_pred, probs = get_c6d_dict(out, grad=False)
        
        # compute loss
        net_out = {'c6d': dict_pred, 'xyz': out.get('xyz', None), 'msa_one_hot': msa_1h, 'lddt': out.get('lddt', None),'alpha': out.get('alpha', None)}
        E_1 = ml.score(net_out)
        
        # Metropolis criteria for each batch 
        T_acc = sf * args.T_acc_0
        acceptances = torch.clamp(torch.exp(-(torch.tensor(E_1-E_0))/T_acc), min=0.0, max=1.0)
        uniform_random = torch.rand(B).to(out_device)
        bool_accept = (uniform_random < acceptances).long()
        E_0 = bool_accept * E_1 + (1-bool_accept) * E_0 
        bool_accept_np = bool_accept.detach().cpu().numpy()
        msa[:,:,:] = bool_accept_np[:,None,None] * mutated_msa[:,:,:] + (1-bool_accept_np[:,None,None]) * msa[:,:,:]
        
        # initial coordinates for next iteration
        if args.reuse_xyz:
            net_kwargs['xyz_prev'] = bool_accept[:,None,None,None]*out['xyz'].detach() + \
                                     (1-bool_accept[:,None,None,None])*net_kwargs['xyz_prev']

        # Every X steps, randomly choose fittest sequences to move forward with
        if (step%200 == 0) and (step != 0):
            # get fresh scores
            msa_ = torch.tensor(msa).long().to(device)
            msa_1h = F.one_hot(msa_,21).float()
            out = Net(msa_, msa_one_hot=msa_1h, **net_kwargs)
            dict_pred, probs = get_c6d_dict(out, grad=False)

            net_out = {'c6d': dict_pred, 'xyz': out.get('xyz', None),'msa_one_hot':msa_1h, 'lddt': out.get('lddt', None),'alpha': out.get('alpha', None)}
            E_1 = ml.score(net_out)
            
            L_ = torch.tensor(L, device=out_device, dtype=E_1.dtype)
            p = (-E_1 / (T_acc * L_.sqrt())).softmax(0) # normalize sd of E_1
            p = p.cpu().numpy()
            sel = np.random.choice(B,B,p=p)
            msa = msa[sel]
            
            if 'prob' in trk:
                trk['prob'].append(p.tolist())
            else:
                trk['prob'] = [p.tolist()]
        
        # track intermediate losses
        if args.track_step is not None and step % (10*args.track_step) == 0:
            trk['step'].append(int(step))
            trk['step_type'].append('mcmc')
            trk['loss_tot'].append(E_0.cpu().tolist())
            for name, value in ml.losses.items():
                trk['loss_'+name].append(value.cpu().tolist())

        # best design so far
        if E_0.min() < trb['loss_tot']:
            i = E_0.argmin()
            trb['loss_tot'] = float(E_0[i])
            trb['msa'] = msa[i][None].copy() # keep singleton batch dim
       
        if step % 5 == 0: 
            print(f'{step:>12}', end='')
            ml.print_losses_line()
            #print(msa_list_AA_mutated)

        # output intermediate result
        out_step = []
        if args.out_interval is not None:
            out_step.extend(np.arange(args.out_interval*10,steps,args.out_interval*10))
        if args.out_step is not None:
            out_step.extend([int(x[1:]) for x in args.out_step.split(',') if x[0]=='m'])
        if step in out_step:
            save_result(f'{trb["out_prefix"]}_g{step}', Net, ml, msa, args, trb, trk, 
                        net_kwargs=net_kwargs)
            if args.drop > 0: Net = enable_dropout(Net) # dropout gets turned off when saving result


    print(f'Max CUDA memory: {torch.cuda.max_memory_allocated()/1e9:.4f}G')
    torch.cuda.reset_peak_memory_stats()

    return msa, out

