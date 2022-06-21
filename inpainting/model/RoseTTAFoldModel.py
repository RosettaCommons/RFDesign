import torch
import torch.nn as nn
from Embeddings import MSA_emb, Extra_emb, Templ_emb, Recycling
from Track_module import IterativeFeatureExtractor
from AuxiliaryPredictor import DistanceNetwork, MaskedTokenNetwork 
from constant import INIT_CRDS

from icecream import ic 
torch.set_printoptions(sci_mode=False)

class RoseTTAFoldModule(nn.Module):
    def __init__(self, n_module_2track=4, n_module_3track=4,\
                 d_msa=256, d_msa_full=64, d_pair=128, d_templ=64,
                 n_head_msa=8, n_head_pair=4, n_head_templ=4,
                 d_hidden=32, d_hidden_templ=64,
                 p_drop=0.15,d_t1d=21+1,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}):
        super(RoseTTAFoldModule, self).__init__()
        #
        self.latent_emb = MSA_emb(d_msa=d_msa, d_pair=d_pair, p_drop=p_drop)
        self.full_emb = Extra_emb(d_msa=d_msa_full, d_init=23, p_drop=p_drop)
        self.templ_emb = Templ_emb(d_pair=d_pair, d_templ=d_templ, n_head=n_head_templ,
                                   d_hidden=d_hidden_templ, p_drop=0.25, d_t1d=d_t1d)

        self.recycle = Recycling(d_msa=d_msa, d_pair=d_pair)
        #
        self.extractor = IterativeFeatureExtractor(n_module_2track=n_module_2track,
                                                   n_module_3track=n_module_3track,
                                                   d_msa=d_msa, d_msa_full=d_msa_full,
                                                   d_pair=d_pair, d_hidden=d_hidden,
                                                   n_head_msa=n_head_msa, n_head_pair=n_head_pair,
                                                   SE3_param=SE3_param,
                                                   p_drop=p_drop)
        ##
        self.c6d_pred = DistanceNetwork(d_pair, p_drop=p_drop)
        self.aa_pred = MaskedTokenNetwork(d_msa, p_drop=p_drop)

    def forward(self, msa_latent, msa_full, seq, idx, t1d=None, t2d=None,
                msa_prev=None, pair_prev=None, xyz_prev=None,
                return_raw=False, return_full=False,
                use_checkpoint=False):
        B, N, L = msa_latent.shape[:3]
        # Get embeddings

        msa_latent, pair = self.latent_emb(msa_latent, seq, idx)
        msa_full = self.full_emb(msa_full, seq, idx)
        #
        if msa_prev == None:
            msa_prev = torch.zeros_like(msa_latent[:,0])
            pair_prev = torch.zeros_like(pair)
            xyz_prev = INIT_CRDS.to(msa_latent.device).reshape(1,1,3,3).expand(B,L,-1,-1)
            #xyz_prev = torch.zeros((B, L, 3, 3), device=msa_latent.device).float()
        msa_recycle, pair_recycle = self.recycle(msa_prev, pair_prev, xyz_prev)
        msa_latent[:,0] = msa_latent[:,0] + msa_recycle.reshape(B,L,-1)
        pair = pair + pair_recycle
        #
        # add template embedding
        pair = self.templ_emb(t1d, t2d, pair, use_checkpoint=use_checkpoint)
        #
        msa, pair, xyz, lddt = self.extractor(msa_latent, msa_full, pair, idx,
                                              use_checkpoint=use_checkpoint)

        if return_full:
            return msa[:,0], pair, xyz.reshape(-1, B, L, 3, 3)[-1], \
                   logits, logits_aa, xyz.reshape(-1, B, L, 3, 3), lddt.view(B, L)

        if return_raw:
            return msa[:,0], pair, xyz.reshape(-1, B, L, 3, 3)[-1]
        #
        logits = self.c6d_pred(pair)
        logits_aa = self.aa_pred(msa)
        return logits, logits_aa, xyz.reshape(-1, B, L, 3, 3), lddt.view(B, L)
