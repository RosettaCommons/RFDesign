import torch
import torch.nn as nn
import torch.nn.functional as F
from Embeddings import MSA_emb, Pair_emb_wo_templ, Pair_emb_w_templ, Templ_emb
from Attention_module_w_str import IterativeFeatureExtractor
from DistancePredictor import DistanceNetwork

class TrunkModule(nn.Module):
    def __init__(self, n_module=4, n_module_str=4, n_layer=4,\
                 d_msa=64, d_pair=128, d_templ=64,\
                 n_head_msa=4, n_head_pair=8, n_head_templ=4,
                 d_hidden=64, r_ff=4, n_resblock=1, p_drop=0.1, 
                 performer_L_opts=None, performer_N_opts=None,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, 
                 use_templ=False):
        super(TrunkModule, self).__init__()
        self.use_templ = use_templ
        #
        self.msa_emb = MSA_emb(d_model=d_msa, p_drop=p_drop, max_len=5000)
        if use_templ:
            self.templ_emb = Templ_emb(d_templ=d_templ, n_att_head=n_head_templ, r_ff=r_ff, 
                                       performer_opts=performer_L_opts, p_drop=0.0)
            self.pair_emb = Pair_emb_w_templ(d_model=d_pair, d_templ=d_templ, p_drop=p_drop)
        else:
            self.pair_emb = Pair_emb_wo_templ(d_model=d_pair, p_drop=p_drop)
        #
        self.feat_extractor = IterativeFeatureExtractor(n_module=n_module,\
                                                        n_module_str=n_module_str,\
                                                        n_layer=n_layer,\
                                                        d_msa=d_msa, d_pair=d_pair, d_hidden=d_hidden,\
                                                        n_head_msa=n_head_msa, \
                                                        n_head_pair=n_head_pair,\
                                                        r_ff=r_ff, \
                                                        n_resblock=n_resblock,
                                                        p_drop=p_drop,
                                                        performer_N_opts=performer_N_opts,
                                                        performer_L_opts=performer_L_opts,
                                                        SE3_param=SE3_param)
        self.c6d_predictor = DistanceNetwork(1, d_pair, p_drop=p_drop)

    def forward(self, msa, idx=None, t1d=None, t2d=None, msa_one_hot=None, 
                use_transf_checkpoint=False, repeat_template=False):
        B, N, L = msa.shape
        seq = msa[:,0]
        
        if repeat_template:
            if idx != None:
                idx = idx.repeat(B,1)
            if t1d != None:
                t1d = t1d.repeat(B,1,1,1)
            if t2d != None:
                t2d = t2d.repeat(B,1,1,1,1)
        
        
        if idx == None:
            idx = torch.arange(L, device=msa.device).unsqueeze(0).expand(B,-1)
        # Get embeddings
        if msa_one_hot != None:
            msa = self.msa_emb(msa, idx, msa_one_hot)
            if self.use_templ:
                if t1d == None: # t1d & t2d as zero matrices
                    t1d = torch.zeros((B, 1, L, 3), device=msa.device).float()
                    t2d = torch.zeros((B, 1, L, L, 10), device=msa.device).float()
                tmpl = self.templ_emb(t1d, t2d, idx)
                pair = self.pair_emb(seq, idx, tmpl, msa_one_hot[:,0])
            else:
                pair = self.pair_emb(seq, idx, msa_one_hot[:,0])
        else:
            msa = self.msa_emb(msa, idx)
            if self.use_templ:
                if t1d == None: # t1d & t2d as zero matrices
                    t1d = torch.zeros((B, 1, L, 3), device=msa.device).float()
                    t2d = torch.zeros((B, 1, L, L, 10), device=msa.device).float()
                tmpl = self.templ_emb(t1d, t2d, idx)
                pair = self.pair_emb(seq, idx, tmpl)
            else:
                pair = self.pair_emb(seq, idx)


        # Extract features
        if msa_one_hot != None:
            seq1hot = msa_one_hot[:,0]
        else:
            seq1hot = F.one_hot(seq,21)


        msa, pair, xyz, lddt = self.feat_extractor(msa, pair, seq1hot, idx,
                                          use_transf_checkpoint=use_transf_checkpoint)

        # Predict 6D coords
        logits = self.c6d_predictor(pair)
       
        return {'dist'  : logits[0],
                'omega' : logits[1],
                'theta' : logits[2],
                'phi'   : logits[3],
                'xyz'   : xyz[-1].view(B,L,3,3),
                'lddt'  : lddt.view(B,L)}
