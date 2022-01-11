import torch
import torch.nn as nn
from Embeddings import MSA_emb, Pair_emb_wo_templ, Pair_emb_w_templ, Templ_emb
from Attention_module_w_str import IterativeFeatureExtractor
from AuxiliaryPredictor import DistanceNetwork, TorsionNetwork, MaskedTokenNetwork

class TrunkModule(nn.Module):
    def __init__(self, n_module=4, n_module_str=4, n_layer=4,\
                 d_msa=64, d_msa_full=64, d_pair=128, d_templ=64,\
                 n_head_msa=4, n_head_pair=8, n_head_templ=4,
                 d_hidden=64, r_ff=4, n_resblock=1, p_drop=0.1, 
                 performer_L_opts=None, performer_N_opts=None,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, 
                 use_templ=False, msa_full_mode='zero', device0=0, device1=0):
        super(TrunkModule, self).__init__()
        self.use_templ = use_templ
        self.device0 = device0
        self.device1 = device1
        self.msa_full_mode = msa_full_mode # either 'zero' (msa_full is all zeros) or 
                                           # 'same' (msa_full is same as msa_latent)
        #
        self.latent_emb = MSA_emb(d_model=d_msa, p_drop=p_drop, max_len=5000).to(device0)
        self.msa_emb = MSA_emb(d_model=d_msa_full, p_drop=p_drop, max_len=5000, incl_query=False).to(device0)
        if use_templ:
            self.templ_emb = Templ_emb(d_templ=d_templ, n_att_head=n_head_templ, r_ff=r_ff, 
                                       performer_opts=performer_L_opts, p_drop=0.0).to(device0)
            self.pair_emb = Pair_emb_w_templ(d_model=d_pair, d_templ=d_templ, p_drop=p_drop).to(device0)
        else:
            self.pair_emb = Pair_emb_wo_templ(d_model=d_pair, p_drop=p_drop).to(device0)
        #
        self.feat_extractor = IterativeFeatureExtractor(n_module=n_module,\
                                                        n_module_str=n_module_str,\
                                                        n_layer=n_layer,\
                                                        d_msa_full=d_msa_full, \
                                                        d_msa=d_msa, d_pair=d_pair, d_hidden=d_hidden,\
                                                        n_head_msa=n_head_msa, \
                                                        n_head_pair=n_head_pair,\
                                                        r_ff=r_ff, \
                                                        n_resblock=n_resblock,
                                                        p_drop=p_drop,
                                                        performer_N_opts=None,
                                                        performer_L_opts=performer_L_opts,
                                                        SE3_param=SE3_param,
                                                        device0=device0,
                                                        device1=device1)
        #
        self.c6d_pred = DistanceNetwork(d_pair, p_drop=p_drop).to(device1)
        self.tor_pred = TorsionNetwork(d_msa, p_drop=p_drop).to(device1)
        self.aa_pred = MaskedTokenNetwork(d_msa, p_drop=p_drop).to(device1)

    def forward(self, msa, seq=None, idx=None, t1d=None, t2d=None, seq1hot=None, msa_one_hot=None,
                use_transf_checkpoint=False):

        B, N, L = msa.shape

        if seq is None:
            seq = msa[:,0]
        if idx is None:
            idx = torch.arange(L, device=msa.device).unsqueeze(0).expand(B,-1)

        if msa_one_hot is not None:
            seq1hot = msa_one_hot[:,0]
            if msa_one_hot.shape[-1] < 22:
                pad_size = 22 - msa_one_hot.shape[-1]
                pad_row = torch.zeros(1,1,L,pad_size).to(msa_one_hot.device)
                msa_one_hot = torch.cat([msa_one_hot, pad_row],-1) # pad to 22 tokens

        if seq1hot is None:
            seq1hot = torch.nn.functional.one_hot(seq, num_classes=21).float()

        # Get embeddings
        msa_latent = self.latent_emb(msa, idx, msa_one_hot=msa_one_hot)
        if self.msa_full_mode == 'zero':
            msa_full_1hot = torch.zeros(B,1,L,22).float().to(msa.device)
        elif self.msa_full_mode == 'same':
            msa_full_1hot = msa_one_hot
        msa_full = self.msa_emb(msa, idx=idx, msa_one_hot=msa_full_1hot)

        if self.use_templ:
            if t1d is None:
                t1d = torch.zeros((B, 1, L, 1), device=msa.device).float()
                t2d = torch.zeros((B, 1, L, L, 46), device=msa.device).float()
            tmpl = self.templ_emb(t1d, t2d, idx)
            pair = self.pair_emb(seq, idx, tmpl, seq1hot)
        else:
            pair = self.pair_emb(seq, idx, seq1hot)
        
        # Extract features
        msa, pair, xyz, lddt = self.feat_extractor(msa_latent, msa_full, pair, seq1hot, idx,
                                     use_transf_checkpoint=use_transf_checkpoint)
        logits = self.c6d_pred(pair)
        logits_tors = self.tor_pred(msa)
        logits_aa = self.aa_pred(msa)
        
        prob_s = list()
        for l in logits:
            prob_s.append(nn.Softmax(dim=1)(l)) # (B, C, L, L)
        prob_s = torch.cat(prob_s, dim=1).permute(0,2,3,1)

        out = {}
        out['dist'] = logits[0]
        out['omega'] = logits[1]
        out['theta'] = logits[2]
        out['phi'] = logits[3]
        out['xyz'] = xyz[-1].view(B,L,3,3)
        out['lddt'] = lddt.view(B, L)
        out['prob_s'] = prob_s
        out['logits_aa'] = logits_aa

        return out
