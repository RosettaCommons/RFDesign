import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import _get_clones, EncoderLayer, Encoder, InterEncoderLayer, InterEncoder, SpecialEncoderLayer, SpecialEncoder
import Transformer
import torch.utils.checkpoint as checkpoint
from resnet import ResidualNetwork

# Attention module based on AlphaFold2's idea written by Minkyung Baek
#  - Iterative MSA feature extraction
#    - 1) MSA2Pair: extract pairwise feature from MSA --> added to previous residue-pair features
#                   architecture design inspired by CopulaNet paper
#    - 2) MSA2MSA:  process MSA features using Transformer (or Performer) encoder. (Attention over L first followed by attention over N)
#    - 3) Pair2MSA: Update MSA features using pair feature
#    - 4) Pair2Pair: process pair features using Transformer (or Performer) encoder.

class MSA2Pair(nn.Module):
    def __init__(self, n_feat=64, n_feat_out=128, n_feat_proj=32,
                 n_resblock=1, p_drop=0.1):
        super(MSA2Pair, self).__init__()
        # project down embedding dimension (n_feat --> n_feat_proj)
        self.proj_1 = nn.Linear(n_feat, n_feat_proj)
        self.norm_2d = nn.LayerNorm(n_feat_proj*n_feat_proj)
        
        # project down to output dimension (pair feature dimension)
        self.proj_2 = nn.Linear(n_feat_proj**2, n_feat_out)
        
        # ResNet to update pair features 
        self.update = ResidualNetwork(n_resblock, n_feat_out*2+n_feat*4, n_feat_out, n_feat_out, p_drop=p_drop)

    def forward(self, msa, pair_orig):
        # Input: MSA embeddings (B, N, L, K), original pair embeddings (B, L, L, C)
        # Output: updated pair info (B, L, L, C)
        B, N, L, _ = msa.shape
        # project down to reduce memory
        x_down = self.proj_1(msa) # (B, N, L, n_feat_proj)
        #pair = torch.einsum('abij,ablm->ailjm', x_down, x_down)
        pair = torch.einsum('abij,ablm->ailjm', x_down, x_down/float(N)) # outer-product & average pool
        pair = pair.reshape(B*L,L,-1) # (B, L, L, n_feat_proj**2)
        pair = self.norm_2d(pair)
        pair = pair.reshape(B, L, L, -1)
        pair = self.proj_2(pair) # (B, L, L, n_feat_out) # project down to pair dimension

        # average pooling over N of given MSA info
        feat_1d = msa.mean(1) # (B,L,K)
        # query sequence info
        query = msa[:,0] # (B,L,K)
        feat_1d = torch.cat((feat_1d, query), dim=-1) # additional 1D features
        # tile 1D features
        left = feat_1d.unsqueeze(2).repeat(1, 1, L, 1)
        right = feat_1d.unsqueeze(1).repeat(1, L, 1, 1)
        # update original pair features through convolutions after concat
        pair = torch.cat((pair_orig, pair, left, right), -1)
        pair = pair.permute(0,3,1,2).contiguous() # prep for convolution layer
        pair = self.update(pair)
        pair = pair.permute(0,2,3,1).contiguous() # (B, L, L, C)

        return pair

class MSA2MSA(nn.Module):
    def __init__(self, n_layer=1, n_att_head=8, n_feat=256, r_ff=4, p_drop=0.1,
                 performer_N_opts=None, performer_L_opts=None):
        super(MSA2MSA, self).__init__()
        # attention along N
        enc_layer_1 = EncoderLayer(d_model=n_feat, d_ff=n_feat*r_ff,
                                   heads=n_att_head, p_drop=p_drop,
                                   performer_opts=performer_N_opts)
        self.encoder_1 = Encoder(enc_layer_1, n_layer, n_feat)
        # attention along L
        enc_layer_2 = EncoderLayer(d_model=n_feat, d_ff=n_feat*r_ff,
                                   heads=n_att_head, p_drop=p_drop,
                                   performer_opts=performer_L_opts)
        self.encoder_2 = Encoder(enc_layer_2, n_layer, n_feat)

    def forward(self, x):
        # Input: MSA embeddings (B, N, L, K)
        # Output: updated MSA embeddings (B, N, L, K)
        B, N, L, _ = x.shape
        # attention along N
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(B*L, N, -1)
        x = self.encoder_1(x)
        x = x.view(B, L, N, -1).permute(0,2,1,3).contiguous()
        # attention along L
        x = x.view(B*N, L, -1)
        x = self.encoder_2(x)
        x = x.view(B, N, L, -1)
        return x

class Pair2MSA(nn.Module):
    def __init__(self, n_layer=1, n_att_head=4, n_feat_in=128, n_feat_out=256, r_ff=4, p_drop=0.1):
        super(Pair2MSA, self).__init__()
        enc_layer = SpecialEncoderLayer(heads=n_att_head, \
                                        d_in=n_feat_in, d_out=n_feat_out,\
                                        d_ff=n_feat_out*r_ff,\
                                        p_drop=p_drop)
        self.encoder = SpecialEncoder(enc_layer, n_layer, n_feat_out)

    def forward(self, pair, msa):
        out = self.encoder(pair, msa) # (B, N, L, K)
        return out

class Pair2Pair(nn.Module):
    def __init__(self, n_layer=1, n_att_head=8, n_feat=128, r_ff=4, p_drop=0.1,
                 performer_L_opts=None):
        super(Pair2Pair, self).__init__()
        enc_layer_1 = EncoderLayer(d_model=n_feat, d_ff=n_feat*r_ff,
                                   heads=n_att_head, p_drop=p_drop,
                                   performer_opts=performer_L_opts)
        self.encoder_1 = Encoder(enc_layer_1, n_layer, n_feat)
        enc_layer_2 = EncoderLayer(d_model=n_feat, d_ff=n_feat*r_ff,
                                   heads=n_att_head, p_drop=p_drop,
                                   performer_opts=performer_L_opts)
        self.encoder_2 = Encoder(enc_layer_2, n_layer, n_feat)
    def forward(self, x):
        # Input: residue pair embeddings (B, L, L, C)
        # Ouput: residue pair embeddings (B, L, L, C)
        # attention over column
        B, L = x.shape[:2]
        x = x.view(B*L, L, -1)
        x = self.encoder_1(x) # attention over column
        x = x.view(B, L, L, -1)
        x = x.permute(0,2,1,3).contiguous().view(B*L, L, -1)
        x = self.encoder_2(x) # attention over row
        x = x.view(B, L, L, -1).permute(0, 2, 1, 3).contiguous()
        return x

class IterBlock(nn.Module):
    def __init__(self, n_layer=1, d_msa=64, d_pair=128, n_head_msa=4, n_head_pair=8, r_ff=4,
                 n_resblock=1, p_drop=0.1, performer_L_opts=None, performer_N_opts=None):
        super(IterBlock, self).__init__()
        
        self.msa2msa = MSA2MSA(n_layer=n_layer, n_att_head=n_head_msa, n_feat=d_msa,
                               r_ff=r_ff, p_drop=p_drop,
                               performer_N_opts=performer_N_opts,
                               performer_L_opts=performer_L_opts)
        self.msa2pair = MSA2Pair(n_feat=d_msa, n_feat_out=d_pair, n_feat_proj=32,
                                 n_resblock=n_resblock, p_drop=p_drop)
        self.pair2pair = Pair2Pair(n_layer=n_layer, n_att_head=n_head_pair,
                                   n_feat=d_pair, r_ff=r_ff, p_drop=p_drop,
                                   performer_L_opts=performer_L_opts)
        self.pair2msa = Pair2MSA(n_layer=n_layer, n_att_head=n_head_pair, 
                                 n_feat_in=d_pair, n_feat_out=d_msa, r_ff=r_ff, p_drop=p_drop)

    def forward(self, msa, pair, use_transf_checkpoint=False):
        # input:
        #   msa: initial MSA embeddings (N, L, d_msa)
        #   pair: initial residue pair embeddings (L, L, d_pair)
        if use_transf_checkpoint:
            Transformer.USE_CHECKPOINT = True
            
        # 1. process MSA features
        msa = self.msa2msa(msa)
        
        # 2. update pair features using given MSA
        pair = self.msa2pair(msa, pair)

        # 3. process pair features
        pair = self.pair2pair(pair)
        
        # 4. update MSA features using updated pair features
        msa = self.pair2msa(pair, msa)
        

        if use_transf_checkpoint:
            Transformer.USE_CHECKPOINT = False
        
        return msa, pair

class IterBlockShare(nn.Module):
    def __init__(self, n_module=4, n_layer=1, d_msa=64, d_pair=128,
                 n_head_msa=4, n_head_pair=8, r_ff=4,
                 n_resblock=1, p_drop=0.1,
                 performer_L_opts=None, performer_N_opts=None):
        super(IterBlockShare, self).__init__()
        self.n_module = n_module 
        self.msa2msa = MSA2MSA(n_layer=n_layer, n_att_head=n_head_msa, n_feat=d_msa,
                               r_ff=r_ff, p_drop=p_drop,
                               performer_N_opts=performer_N_opts,
                               performer_L_opts=performer_L_opts)
        self.msa2pair = MSA2Pair(n_feat=d_msa, n_feat_out=d_pair, n_feat_proj=32,
                                 n_resblock=n_resblock, p_drop=p_drop)
        self.pair2pair = Pair2Pair(n_layer=n_layer, n_att_head=n_head_pair,
                                   n_feat=d_pair, r_ff=r_ff, p_drop=p_drop,
                                   performer_L_opts=performer_L_opts)
        self.pair2msa = Pair2MSA(n_layer=n_layer, n_att_head=n_head_pair, 
                                 n_feat_in=d_pair, n_feat_out=d_msa, r_ff=r_ff, p_drop=p_drop)

    def forward(self, msa, pair):
        # input:
        #   msa: initial MSA embeddings (N, L, d_msa)
        #   pair: initial residue pair embeddings (L, L, d_pair)
        
        for i_m in range(self.n_module):
            # 1. process MSA features
            msa = self.msa2msa(msa)
            
            # 2. update pair features using given MSA
            pair = self.msa2pair(msa, pair)
            
            # 3. process pair features
            pair = self.pair2pair(pair)

            # 4. update MSA features using updated pair features
            msa = self.pair2msa(pair, msa)

        return msa, pair

class FinalBlock(nn.Module):
    def __init__(self, n_layer=1, d_msa=64, d_pair=128, n_head_msa=4, r_ff=4,
                 n_resblock=1, p_drop=0.1, performer_L_opts=None, performer_N_opts=None):
        super(FinalBlock, self).__init__()
        
        self.msa2msa = MSA2MSA(n_layer=n_layer, n_att_head=n_head_msa, n_feat=d_msa,
                               r_ff=r_ff, p_drop=p_drop,
                               performer_N_opts=performer_N_opts,
                               performer_L_opts=performer_L_opts)
        self.msa2pair = MSA2Pair(n_feat=d_msa, n_feat_out=d_pair, n_feat_proj=32,
                                 n_resblock=n_resblock, p_drop=p_drop)

    def forward(self, msa, pair, use_transf_checkpoint=False):
        # input:
        #   msa: initial MSA embeddings (N, L, d_msa)
        #   pair: initial residue pair embeddings (L, L, d_pair)
        if use_transf_checkpoint:
            Transformer.USE_CHECKPOINT = True
            
        # 1. process MSA features
        msa = self.msa2msa(msa)
        
        # 2. update pair features using given MSA
        pair = self.msa2pair(msa, pair)

        if use_transf_checkpoint:
            Transformer.USE_CHECKPOINT = False
        
        return msa, pair

class IterativeFeatureExtractor(nn.Module):
    def __init__(self, n_module=4, n_diff_module=2, n_layer=4, d_msa=256, d_pair=128,
                 n_head_msa=8, n_head_pair=8, r_ff=4, 
                 n_resblock=1, p_drop=0.1,
                 performer_L_opts=None, performer_N_opts=None):
        super(IterativeFeatureExtractor, self).__init__()
        self.n_module = n_module
        self.n_diff_module = n_diff_module
        self.n_share_module = n_module - n_diff_module
        #
        self.initial = Pair2Pair(n_layer=n_layer, n_att_head=n_head_pair,
                                 n_feat=d_pair, r_ff=r_ff, p_drop=p_drop,
                                 performer_L_opts=performer_L_opts)

        if self.n_diff_module > 0:
            self.iter_block_1 = _get_clones(IterBlock(n_layer=n_layer, 
                                                      d_msa=d_msa, d_pair=d_pair,
                                                      n_head_msa=n_head_msa,
                                                      n_head_pair=n_head_pair,
                                                      r_ff=r_ff,
                                                      n_resblock=n_resblock,
                                                      p_drop=p_drop,
                                                      performer_N_opts=performer_N_opts,
                                                      performer_L_opts=performer_L_opts
                                                      ), n_diff_module)
        if self.n_share_module > 0:
            self.iter_block_2 = IterBlockShare(n_module=n_module-n_diff_module, n_layer=n_layer,
                                               d_msa=d_msa, d_pair=d_pair,
                                               n_head_msa=n_head_msa, n_head_pair=n_head_pair,
                                               r_ff=r_ff,
                                               n_resblock=n_resblock,
                                               p_drop=p_drop,
                                               performer_N_opts=performer_N_opts,
                                               performer_L_opts=performer_L_opts
                                               )
        
        self.final = FinalBlock(n_layer=n_layer, d_msa=d_msa, d_pair=d_pair,
                               n_head_msa=n_head_msa, r_ff=r_ff,
                               n_resblock=n_resblock, p_drop=p_drop,
                               performer_L_opts=performer_L_opts, performer_N_opts=performer_N_opts)

    def forward(self, msa, pair, use_block_checkpoint=False, use_transf_checkpoint=False):
        if use_block_checkpoint:
            # for gradient checkpointing
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(inputs[0], inputs[1])
                return custom_forward
        # input:
        #   msa: initial MSA embeddings (N, L, d_msa)
        #   pair: initial residue pair embeddings (L, L, d_pair)
        
        pair_s = list()
        pair = self.initial(pair)
        if self.n_diff_module > 0:
            for i_m in range(self.n_diff_module):
                # extract features from MSA & update original pair features
                if use_block_checkpoint and (not use_transf_checkpoint):
                    msa, pair = checkpoint.checkpoint(create_custom_forward(self.iter_block_1[i_m]), msa, pair) 
                else:
                    msa, pair = self.iter_block_1[i_m](msa, pair, \
                                use_transf_checkpoint=use_transf_checkpoint)
        
        if self.n_share_module > 0:
            if use_block_checkpoint:
                msa, pair = checkpoint.checkpoint(create_custom_forward(self.iter_block_2), msa, pair)
            else:
                msa, pair = self.iter_block_2(msa, pair)
        
        msa, pair = self.final(msa, pair)

        return msa, pair
