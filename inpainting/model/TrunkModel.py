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
                 use_templ=False):
        super(TrunkModule, self).__init__()
        self.use_templ = use_templ
        #
        self.latent_emb = MSA_emb(d_model=d_msa, p_drop=p_drop, max_len=5000)
        self.msa_emb = MSA_emb(d_model=d_msa_full, p_drop=p_drop, max_len=5000, incl_query=False)
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
                                                        d_msa_full=d_msa_full, \
                                                        d_msa=d_msa, d_pair=d_pair, d_hidden=d_hidden,\
                                                        n_head_msa=n_head_msa, \
                                                        n_head_pair=n_head_pair,\
                                                        r_ff=r_ff, \
                                                        n_resblock=n_resblock,
                                                        p_drop=p_drop,
                                                        performer_N_opts=None,
                                                        performer_L_opts=performer_L_opts,
                                                        SE3_param=SE3_param)
        #
        self.c6d_pred = DistanceNetwork(d_pair, p_drop=p_drop)
        self.tor_pred = TorsionNetwork(d_msa, p_drop=p_drop) 
        self.aa_pred = MaskedTokenNetwork(d_msa, p_drop=p_drop)

    def forward(self, msa_latent, msa_full, seq, idx, t1d=None, t2d=None,
                use_transf_checkpoint=False):
        B, N, L = msa_latent.shape
        # Get embeddings
        msa_latent = self.latent_emb(msa_latent, idx)
        msa_full = self.msa_emb(msa_full, idx)
        if self.use_templ:
            tmpl = self.templ_emb(t1d, t2d, idx)
            pair = self.pair_emb(seq, idx, tmpl)
        else:
            pair = self.pair_emb(seq, idx)
        #
        # Extract features
        seq1hot = torch.nn.functional.one_hot(seq, num_classes=21).float()
        msa, pair, xyz, lddt = self.feat_extractor(msa_latent, msa_full, pair, seq1hot, idx,
                                     use_transf_checkpoint=use_transf_checkpoint)
        logits = self.c6d_pred(pair)
        logits_tors = self.tor_pred(msa)
        logits_aa = self.aa_pred(msa)
        
        return logits, logits_aa, logits_tors, xyz.reshape(-1, B, L, 3, 3), lddt.view(B, L)
