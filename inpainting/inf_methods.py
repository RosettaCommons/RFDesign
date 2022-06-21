# module for autofold2 inference methods 
import torch 

def classic_inference(model, msa, msa_extra, seq, t1d, t2d, idx_pdb, N_cycle):
    """
    Classic RF2 inference 
    """
    msa_prev  = None
    pair_prev = None
    xyz_prev  = None 


    for i_cycle in range(N_cycle-1): 
        print('on cycle',i_cycle)
        
        msa_prev, pair_prev, xyz_prev = model(msa[:,i_cycle],
                                              msa_extra[:,i_cycle],
                                                seq[:,i_cycle], idx_pdb,
                                                t1d=t1d, t2d=t2d,
                                                msa_prev=msa_prev, 
                                                pair_prev=pair_prev, 
                                                xyz_prev=xyz_prev, 
                                                return_raw=True, 
                                                use_checkpoint=False)

    i_cycle = N_cycle-1

    logit_s, logit_aa_s, pred_crds, pred_lddts = model(msa[:,i_cycle],
                                                       msa_extra[:,i_cycle], 
                                                       seq[:,i_cycle], idx_pdb,
                                                       t1d=t1d, t2d=t2d, 
                                                       msa_prev=msa_prev, 
                                                       pair_prev=pair_prev, 
                                                       xyz_prev=xyz_prev,
                                                       use_checkpoint=False)
        

    # get sequence by argmaxing final logits  
    seq_out = torch.argmax(logit_aa_s, dim=1)


    return logit_s, logit_aa_s, pred_crds, pred_lddts, seq_out 

