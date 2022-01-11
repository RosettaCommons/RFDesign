# Module containing wrappers for different prediction styles with RosettaFold 
import os, sys, copy
import numpy as np
import torch 
from kinematics import xyz_to_t2d
import pred_util, dj_parser
from dj_util import SampledMask

# absolute path to folder containing this file
script_dir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(script_dir+'/model')
import util 

#from icecream import ic 
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def autoregressive(model, item, order='left'):
    """
    Makes a prediction on item in an autoregressive manner
    
    Parameters:
        model (torch.nn.module, required): RosettaFold model 

        item (dict, required): dictionary containing all the inputs 

        window (tuple, required): beginning and end of window to make predictions in.
                              NOTE: Will loop through entire protein for as many frames as 
                                    fit 

    """
    msa         = item['msa']
    msa_full    = item['msa_full']
    seq         = item['seq']
    idx         = item['idx']
    t1d         = item['t1d']
    t2d         = item['t2d']
    xyz_t       = item['xyz_t']
    device      = item['device']
    t0d         = item['t0d'].to(device)
    args        = item['args']
    mask_seq    = item['mask_seq']
    mask_str    = item['mask_str']
    sm          = item['sm']

    L_binder = (sm.hal_pdb_ch=='A').sum()

    if args.mask_query:
        idx_all = np.arange(len(mask_seq)) # sample entire sequence
    else:
        idx_all = np.where(mask_seq | mask_str)[0] # sample just masked positions

    # print mask string (no receptor)
    print('Autoregressive mask: ')
    print(f'    '+''.join([str(int(i)) for i in (mask_str | mask_seq)[:L_binder]]))

    for i in idx_all: # i = position to fill

        logit_s,\
        logit_aa_s,\
        logit_tor_s,\
        pred_crds,\
        pred_lddts = model(msa, msa_full, seq, idx, t1d, t2d)

        # recycle sequence for next iter
        prob = torch.nn.functional.softmax(logit_aa_s[:,:20,i]/args.autoreg_T, dim=1)
        aa = torch.multinomial(prob, 1)              # sample AA
        seq[:,i]          = aa
        msa[:,...,0,i]      = aa
        msa_full[:,...,0,i] = aa

    return logit_s, logit_aa_s, logit_tor_s, pred_crds, pred_lddts, seq.squeeze().detach().cpu().numpy()

def one_shot(model, item):
    """
    Makes prediction on item in a one-shot argmax manner 
    
    Parameters:
        model (torch.nn.module, required): RosettaFold model 

        item (dict, required): dictionary containing all the inputs 

    """
    # simple forward pass 
    logit_s, logit_aa_s, logit_tor_s, pred_crds, pred_lddts = \
        model(item['msa'], item['msa_full'], item['seq'], item['idx'], 
              t1d=item['t1d'], t2d=item['t2d'])

    seq = logit_aa_s.argmax(1).squeeze().detach().cpu().numpy() # assumes batch=1 and removes that dimension

    return logit_s, logit_aa_s, logit_tor_s, pred_crds, pred_lddts, seq



def multi_shot(model, item):
    """
    Iteratively make predictions on masked region for however many iters you want 

    Parameters:
        
        model (torch.nn.module, required): RosettaFold model 

        item (dict, required): dictionary containing all inputs 

        w_start (int, required): Start of the prediction window

        w_end (int, required): End of the prediction window 

        iters (int, optional): How many iterations we would like to make 

        sample (bool, optional): Whether to sample sequence from logits or argmax. 
                                 If True, will sample. Else argmax. 
    """
    msa         = item['msa']      # (B,N,L)
    msa_full    = item['msa_full'] # (B,N_full,L)
    seq         = item['seq']      # (B,L)
    idx         = item['idx']      # (B,L)
    xyz_t       = item['xyz_t']    # (B,T,L,3,3)
    t0d         = item['t0d']      # (B,T,3)
    t1d         = item['t1d']      # (B,T,L,1)
    t2d         = item['t2d']      # (B,T,L,L,46)
    mask_str    = item['mask_str']     # (L,)
    mask_seq    = item['mask_seq']     # (L,)
    sm          = item['sm']
    sm_motif    = item['sm_motif']
    args        = item['args']
    device      = item['device']

    L_binder = (sm.hal_pdb_ch=='A').sum()

    # set up template inputs
    if args.native_motif_only:
        t0d_immutable = t0d[:,:args.native_motif_repeat]
        t1d_immutable = t1d[:,:args.native_motif_repeat]
        xyz_immutable = xyz_t[:,:args.native_motif_repeat]
    else:
        t0d_immutable = t0d
        t1d_immutable = t1d
        xyz_immutable = xyz_t
    
    t2d = xyz_to_t2d(xyz_t, t0d.detach())
    
    # random walk
    mask_orig=mask_str
    in_rw=0
    del_rw=0
    
    if args.make_ensemble_random_walk:
        print('Random walk mode')

    for i in range(args.n_iters):        
        if args.make_ensemble_random_walk:
            mask = mask_str
            #################################################
            #subsample mask and apply to inputs
            ######################################################
            if args.window is not None or args.mask is not None or args.contigs is not None:
                k = min(args.k_ensemble, len(np.where(mask_orig)[0]))
                idx_submask = np.random.choice(np.where(mask_orig)[0], k, replace=False)
                #print(np.where(mask_orig)[0])
                #print(idx_submask)
                mask = np.full_like(mask, False)
                mask[idx_submask] = True # subsampled mask
            ############################################################    
            #option for indel one residue at a time up to a max len#
            ############################################################
            if args.indel_rw and (in_rw < args.in_rw or del_rw < args.del_rw):
                choice=np.random.choice(['ins','keep','del'], 1)
                print(choice)
                if choice=='ins' and in_rw < args.in_rw: 
                    ############################
                    #add on to someplace in seq#
                    ############################
                    mask_orig,xyz_t,seq,msa,\
                    msa_full,mask,idx_ins=pred_util.rw_ins_input(args,mask_orig,xyz_t,seq,msa,msa_full,mask,DEVICE)
                    print(idx_ins)
                    if i !=0:
                        old_mask = np.insert(old_mask, idx_ins, True, axis=0) 
                    t1d_new      = torch.full((1, t1d.shape[1], 1, 1),0).to(DEVICE) 
                    t1d          = torch.cat([t1d[:,:,:idx_ins],t1d_new,t1d[:,:,idx_ins:]],2)
                    idx          = torch.tensor(np.arange(len(mask))[None]).to(DEVICE)
                    in_rw +=1
                elif choice=='del' and del_rw < args.del_rw:
                    #######################
                    #del somemplace in seq#
                    #######################
                    mask_orig,xyz_t,seq,msa,\
                    msa_full,mask,idx_del=pred_util.rw_del_input(args,mask_orig,xyz_t,seq,msa,msa_full,mask,DEVICE)
                    print(idx_del)
                    if i !=0:
                        old_mask = np.delete(old_mask, idx_del, axis=0) 
                    t1d          = torch.cat([t1d[:,:,:idx_del],t1d[:,:,idx_del+1:]],2)
                    idx          = torch.tensor(np.arange(len(mask))[None]).to(DEVICE)
                    del_rw +=1
            # apply mask to network inputs
            if i!=0:
                #refill old template w/ 0.1 b/c sequence was gotten from there so we want some confidence 
                t1d[:,:,old_mask,:] = torch.clamp(t1d[:,:,old_mask,:], min=0, max=args.clamp_1d)  
            old_mask=mask
            seq, msa, msa_full, xyz_t, f1d_t = pred_util.mask_inputs(args, mask, seq[0], msa[0], msa_full[0], xyz_t, t1d[0])
            t2d      = xyz_to_t2d(xyz_t, t0d)#.to(DEVICE)
            t1d      = f1d_t[None]#.to(DEVICE)
            msa      = msa[None,...]#.to(DEVICE)
            msa_full = msa_full[None,...]#.to(DEVICE)
            seq      = seq[None,...]

            # output mask string (no receptor)
            print(f'{i:<4}'+''.join([str(int(i)) for i in mask[:L_binder]]))

        logit_s,\
        logit_aa_s,\
        logit_tor_s,\
        pred_crds,\
        pred_lddts = model(msa, msa_full, seq, idx, t1d, t2d)

        seq_str = ''.join([util.aa_N_1[a] for a in logit_aa_s.argmax(dim=1).squeeze().detach().cpu().numpy()])
        print(f'{i:<4}'+seq_str[:L_binder])

        # recycle sequence information 
        if args.recycle_seq:
            max_seq = logit_aa_s.argmax(dim=1)              # argmax down the AA dim 
            seq[:,mask_seq]        = max_seq[:,mask_seq]
            msa[:,:,mask_seq]      = max_seq[:,mask_seq]
            msa_full[:,:,mask_seq] = max_seq[:,mask_seq]

        ## Prepare template features from structure predictions
        # Use predicted residue-wise structure confidence (lDDT) as 1d template features
        t1d_new = pred_lddts[None,:,:,None].detach()  # input plddt as 1D template confidence 
        t0d_new = t1d_new.mean(dim=-2).expand(-1,-1,3) # mean plddt as scalar template feature
        
        if args.clamp_1d:
            t0d_new = torch.clamp(t0d_new, min=0, max=args.clamp_1d)

            # clamp the 1D template features between 0 and desired value 
            if not args.local_clamp:
                #print('Doing GLOBAL input 1D feature clamp')
                t1d_new = torch.clamp(t1d_new, min=0, max=args.clamp_1d)
            else:
                #print('Doing LOCAL input 1D feature clamp')
                t1d_new[:,:,mask_str,:] = torch.clamp(t1d_new[:,:,mask_str,:], min=0, max=args.clamp_1d)

        # use predicted structure as template coordinates 
        xyz_t_new = pred_crds[-1][None,...].detach() 
        xyz_t_new = xyz_t_new.repeat(1,args.repeat_template,1,1,1)
        
        ## set up structure recycling
        if args.recycle_str_mode == 'pred':
            # replace original template features with new ones
            t0d = t0d_new
            t1d = t1d_new
            xyz_t = xyz_t_new
        elif args.recycle_str_mode == 'both':
            if args.make_ensemble_random_walk and not args.no_template_accum:
                t0d = torch.cat([t0d, t0d_new], dim=1)
                t1d = torch.cat([t1d, t1d_new], dim=1)
                xyz_t = torch.cat([xyz_t, xyz_t_new], dim=1)
            else:
                t0d = torch.cat([t0d_immutable, t0d_new], dim=1)
                t1d = torch.cat([t1d_immutable, t1d_new], dim=1)
                xyz_t = torch.cat([xyz_immutable, xyz_t_new], dim=1)
        
        # downweight native motif region in non-motif templates
        if args.natmotif_region_weight is not None:
            con_hal_idx0 = sm_motif.mappings['con_hal_idx0']
            t1d[:,args.native_motif_repeat:,con_hal_idx0] = max(args.natmotif_region_weight, 0)
            if args.natmotif_region_weight == -1:
                xyz_t[:,args.native_motif_repeat:,con_hal_idx0] = np.nan

        t2d = xyz_to_t2d(xyz_t, t0d.detach())

    item['xyz_t_final'] = xyz_t
    item['t0d_final'] = t0d
    item['t1d_final'] = t1d
    item['t2d_final'] = t2d

    # argmax sequence; assumes batch=1 and removes that dimension
    seq = logit_aa_s.argmax(1).squeeze().detach().cpu().numpy() 

    # restore mask to its original
    item['mask_str'] = mask_orig

    return logit_s, logit_aa_s, logit_tor_s, pred_crds, pred_lddts, seq
        

def msar(model, item):
    """
    Multi-shot autoregressive: first designs structure by multishot, then
    designs sequence by autoregressive.
    """
    args = item['args']
    item2 = copy.deepcopy(item)

    # make prediction
    with torch.no_grad():
        logit_s, logit_aa_s, logit_tor_s, pred_crds, pred_lddts, pred_seq = \
            multi_shot(model, item)

    if args.make_ensemble_random_walk:
        # need to do masking using full mask
        seq, msa, msa_full, xyz_t, f1d_t = pred_util.mask_inputs(args, 
                                                                 item['mask_str'], 
                                                                 item['seq'][0], 
                                                                 item['msa'][0], 
                                                                 item['msa_full'][0], 
                                                                 item['xyz_t'], 
                                                                 item['t1d_final'][0])
        item2['xyz_t']    = xyz_t
        item2['t2d']      = item['t2d_final']
        item2['t1d']      = f1d_t[None]#.to(DEVICE)
        item2['msa']      = msa[None,...]#.to(DEVICE)
        item2['msa_full'] = msa_full[None,...]#.to(DEVICE)
        item2['seq']      = seq[None,...]
    else:
        item2['t1d'] = item['t1d_final']
        item2['t2d'] = item['t2d_final']

    return autoregressive(model, item2)

def confidence(model, item, window, topk=1, mode='des', confidence_metric='probability'):
    """
    Fill up an indow window with predictions by choosing most confident options first

    Parameters:
        
    """

