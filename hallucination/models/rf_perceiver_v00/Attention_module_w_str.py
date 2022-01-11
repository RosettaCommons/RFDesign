import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import *
from Transformer import _get_clones
import Transformer
import torch.utils.checkpoint as checkpoint
from resnet import ResidualNetwork
from SE3_network import SE3Transformer
from InitStrGenerator import InitStr_Network, get_seqsep
import dgl

# Attention module based on AlphaFold2's idea written by Minkyung Baek
#  - Iterative MSA feature extraction
#    - 1) MSA2Pair: extract pairwise feature from MSA --> added to previous residue-pair features
#                   architecture design inspired by CopulaNet paper
#    - 2) MSA2MSA:  process MSA features using Transformer (or Performer) encoder. (Attention over L first followed by attention over N)
#    - 3) Pair2MSA: Update MSA features using pair feature
#    - 4) Pair2Pair: process pair features using Transformer (or Performer) encoder.

def make_graph(xyz, pair, idx, top_k=64, kmin=9):
    '''
    Input:
        - xyz: current backbone cooordinates (B, L, 3, 3)
        - pair: pair features from Trunk (B, L, L, E)
        - idx: residue index from ground truth pdb
    Output:
        - G: defined graph
    '''

    B, L = xyz.shape[:2]
    device = xyz.device
    
    # distance map from current CA coordinates
    D = torch.cdist(xyz, xyz) + torch.eye(L, device=device).unsqueeze(0)*999.9  # (B, L, L)
    # seq sep
    sep = idx[:,None,:] - idx[:,:,None]
    sep = sep.abs() + torch.eye(L, device=device).unsqueeze(0)*999.9
    
    # get top_k neighbors
    D_neigh, E_idx = torch.topk(D, min(top_k, L), largest=False) # shape of E_idx: (B, L, top_k)
    topk_matrix = torch.zeros((B, L, L), device=device)
    topk_matrix.scatter_(2, E_idx, 1.0)

    # put an edge if any of the 3 conditions are met:
    #   1) |i-j| <= kmin (connect sequentially adjacent residues)
    #   2) top_k neighbors
    cond = torch.logical_or(topk_matrix > 0.0, sep < kmin)
    b,i,j = torch.where(cond)
   
    src = b*L+i
    tgt = b*L+j
    G = dgl.graph((src, tgt), num_nodes=B*L).to(device)
    G.edata['d'] = (xyz[b,j,:] - xyz[b,i,:]).detach() # no gradient through basis function
    G.edata['w'] = pair[b,i,j]

    return G 

def rbf(D):
    # Distance radial basis function
    D_min, D_max, D_count = 0., 20., 36
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu[None,:]
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

class Perceiver(nn.Module):
    def __init__(self, n_layer=1, n_att_head=8,
                 n_feat=64, r_ff=4, d_kv=64,
                 performer_opts=None, p_drop=0.1):
        super(Perceiver, self).__init__()
        enc_layer = CrossEncoderLayer(n_feat, n_feat*r_ff, n_att_head, d_kv, d_kv,
                                      performer_opts=performer_opts, p_drop=p_drop)
        self.encoder = CrossEncoder(enc_layer, n_layer) 
    def forward(self, msa_latent, msa_full):
        B, N1, L1 = msa_latent.shape[:3]
        B, N2, L2 = msa_full.shape[:3]
        msa_full = msa_full.permute(0,2,1,3).reshape(B*L2, N2, -1)
        msa_latent = msa_latent.permute(0,2,1,3).reshape(B*L1, N1, -1)
        msa_latent = self.encoder(msa_full, msa_latent)
        msa_latent = msa_latent.reshape(B, L1, N1, -1).permute(0,2,1,3).contiguous()
        return msa_latent

class CoevolExtractor(nn.Module):
    def __init__(self, n_feat_proj, n_feat_out, p_drop=0.1):
        super(CoevolExtractor, self).__init__()
        
        # project down to output dimension (pair feature dimension)
        self.proj_2 = nn.Linear(n_feat_proj**2, n_feat_out)
    def forward(self, x_down, x_down_w):
        B, N, L = x_down.shape[:3]
        
        pair = torch.einsum('abij,ablm->ailjm', x_down, x_down_w) # outer-product & average pool
        pair = pair.reshape(B, L, L, -1)
        pair = self.proj_2(pair) # (B, L, L, n_feat_out) # project down to pair dimension
        return pair

class MSAStr2Pair(nn.Module):
    def __init__(self, n_feat=64, n_feat_out=128, n_feat_proj=24,
                 n_resblock=1, p_drop=0.1, n_att_head=8):
        super(MSAStr2Pair, self).__init__()
        # project down embedding dimension (n_feat --> n_feat_proj)
        self.norm_1 = LayerNorm(n_feat)
        self.proj_1 = nn.Linear(n_feat, n_feat_proj)
        
        self.encoder = SequenceWeight(n_feat_proj, 1, dropout=p_drop)
        self.coevol = CoevolExtractor(n_feat_proj, n_feat_out)

        # ResNet to update pair features 
        self.norm_down = LayerNorm(n_feat_proj)
        self.norm_orig = LayerNorm(n_feat_out)
        self.norm_new  = LayerNorm(n_feat_out)
        self.update = ResidualNetwork(n_resblock, 36+n_feat_out*2+n_feat_proj*4+n_att_head, n_feat_out, n_feat_out, p_drop=p_drop)

    def forward(self, msa, pair_orig, att, xyz):
        # Input: MSA embeddings (B, N, L, K), original pair embeddings (B, L, L, C)
        # Output: updated pair info (B, L, L, C)
        B, N, L, _ = msa.shape
        # project down to reduce memory
        msa = self.norm_1(msa)
        x_down = self.proj_1(msa) # (B, N, L, n_feat_proj)
        
        # get sequence weight
        x_down = self.norm_down(x_down)
        w_seq = self.encoder(x_down).reshape(B, L, 1, N).permute(0,3,1,2)
        feat_1d = w_seq*x_down
        
        if Transformer.USE_CHECKPOINT:
            pair = checkpoint.checkpoint(create_custom_forward(self.coevol), x_down, feat_1d)
        else:
            pair = self.coevol(x_down, feat_1d)

        # average pooling over N of given MSA info
        feat_1d = feat_1d.sum(1)
        
        # query sequence info
        query = x_down[:,0] # (B,L,K)
        feat_1d = torch.cat((feat_1d, query), dim=-1) # additional 1D features
        # tile 1D features
        left = feat_1d.unsqueeze(2).repeat(1, 1, L, 1)
        right = feat_1d.unsqueeze(1).repeat(1, L, 1, 1)
        # update original pair features through convolutions after concat
        pair_orig = self.norm_orig(pair_orig)
        pair = self.norm_new(pair)
        # get rbf feature from given structure
        rbf_feat = rbf(torch.cdist(xyz, xyz))
        pair = torch.cat((rbf_feat, pair_orig, pair, left, right, att), -1)
        pair = pair.permute(0,3,1,2).contiguous() # prep for convolution layer
        pair = self.update(pair)
        pair = pair.permute(0,2,3,1).contiguous() # (B, L, L, C)

        return pair

class MSA2Pair(nn.Module):
    def __init__(self, n_feat=64, n_feat_out=128, n_feat_proj=24,
                 n_resblock=1, p_drop=0.1, n_att_head=8):
        super(MSA2Pair, self).__init__()
        # project down embedding dimension (n_feat --> n_feat_proj)
        self.norm_1 = LayerNorm(n_feat)
        self.proj_1 = nn.Linear(n_feat, n_feat_proj)
        
        self.encoder = SequenceWeight(n_feat_proj, 1, dropout=p_drop)
        self.coevol = CoevolExtractor(n_feat_proj, n_feat_out)

        # ResNet to update pair features 
        self.norm_down = LayerNorm(n_feat_proj)
        self.norm_orig = LayerNorm(n_feat_out)
        self.norm_new  = LayerNorm(n_feat_out)
        self.update = ResidualNetwork(n_resblock, n_feat_out*2+n_feat_proj*4+n_att_head, n_feat_out, n_feat_out, p_drop=p_drop)

    def forward(self, msa, pair_orig, att):
        # Input: MSA embeddings (B, N, L, K), original pair embeddings (B, L, L, C)
        # Output: updated pair info (B, L, L, C)
        B, N, L, _ = msa.shape
        # project down to reduce memory
        msa = self.norm_1(msa)
        x_down = self.proj_1(msa) # (B, N, L, n_feat_proj)
        
        # get sequence weight
        x_down = self.norm_down(x_down)
        w_seq = self.encoder(x_down).reshape(B, L, 1, N).permute(0,3,1,2)
        feat_1d = w_seq*x_down
        
        if Transformer.USE_CHECKPOINT:
            pair = checkpoint.checkpoint(create_custom_forward(self.coevol), x_down, feat_1d)
        else:
            pair = self.coevol(x_down, feat_1d)

        # average pooling over N of given MSA info
        feat_1d = feat_1d.sum(1)
        
        # query sequence info
        query = x_down[:,0] # (B,L,K)
        feat_1d = torch.cat((feat_1d, query), dim=-1) # additional 1D features
        # tile 1D features
        left = feat_1d.unsqueeze(2).repeat(1, 1, L, 1)
        right = feat_1d.unsqueeze(1).repeat(1, L, 1, 1)
        # update original pair features through convolutions after concat
        pair_orig = self.norm_orig(pair_orig)
        pair = self.norm_new(pair)
        pair = torch.cat((pair_orig, pair, left, right, att), -1)
        pair = pair.permute(0,3,1,2).contiguous() # prep for convolution layer
        pair = self.update(pair)
        pair = pair.permute(0,2,3,1).contiguous() # (B, L, L, C)

        return pair

class MSA2MSA(nn.Module):
    def __init__(self, n_layer=1, n_att_head=8, n_feat=256, r_ff=4, p_drop=0.1,
                 performer_N_opts=None, performer_L_opts=None):
        super(MSA2MSA, self).__init__()
        # Axial attention. Soft-tied attention over residue
        enc_layer = AxialEncoderLayer(d_model=n_feat, d_ff=n_feat*r_ff,
                                      heads=n_att_head, p_drop=p_drop,
                                      performer_opts=performer_N_opts,
                                      use_soft_row=True)
        self.encoder = Encoder(enc_layer, n_layer)

    def forward(self, x):
        # Input: MSA embeddings (B, N, L, K)
        # Output: updated MSA embeddings (B, N, L, K)
        x, att = self.encoder(x, return_att=True)
        return x, att

class Pair2MSA(nn.Module):
    def __init__(self, n_layer=1, n_att_head=4, n_feat_in=128, n_feat_out=256, r_ff=4, p_drop=0.1):
        super(Pair2MSA, self).__init__()
        enc_layer = DirectEncoderLayer(heads=n_att_head, \
                                       d_in=n_feat_in, d_out=n_feat_out,\
                                       d_ff=n_feat_out*r_ff,\
                                       p_drop=p_drop)
        self.encoder = CrossEncoder(enc_layer, n_layer)

    def forward(self, pair, msa):
        out = self.encoder(pair, msa) # (B, N, L, K)
        return out

class Pair2Pair(nn.Module):
    def __init__(self, n_layer=1, n_att_head=8, n_feat=128, r_ff=4, p_drop=0.1,
                 performer_L_opts=None):
        super(Pair2Pair, self).__init__()
        enc_layer = AxialEncoderLayer(d_model=n_feat, d_ff=n_feat*r_ff,
                                      heads=n_att_head, p_drop=p_drop,
                                      performer_opts=performer_L_opts)
        self.encoder = Encoder(enc_layer, n_layer)
    
    def forward(self, x):
        return self.encoder(x)

class Str2Str(nn.Module):
    def __init__(self, d_msa=64, d_pair=128, d_state=16, 
            SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, p_drop=0.1):
        super(Str2Str, self).__init__()
        
        # initial node & pair feature process
        self.norm_msa = LayerNorm(d_msa)
        self.norm_pair = LayerNorm(d_pair)
        self.norm_state = LayerNorm(d_state)
        self.encoder_seq = SequenceWeight(d_msa, 1, dropout=p_drop)
    
        self.embed_x = nn.Linear(d_msa+21+d_state, SE3_param['l0_in_features'])
        self.embed_e1 = nn.Linear(d_pair, SE3_param['num_edge_features'])
        self.embed_e2 = nn.Linear(SE3_param['num_edge_features']+36+1, SE3_param['num_edge_features'])
        
        self.norm_node = LayerNorm(SE3_param['l0_in_features'])
        self.norm_edge1 = LayerNorm(SE3_param['num_edge_features'])
        self.norm_edge2 = LayerNorm(SE3_param['num_edge_features'])
        
        self.se3 = SE3Transformer(**SE3_param)
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, msa, pair, xyz, state, seq1hot, idx, top_k=64, eps=1e-5):
        # process msa & pair features
        B, N, L = msa.shape[:3]
        msa = self.norm_msa(msa)
        pair = self.norm_pair(pair)
        state = self.norm_state(state)
       
        w_seq = self.encoder_seq(msa).reshape(B, L, 1, N).permute(0,3,1,2)
        msa = w_seq*msa
        msa = msa.sum(dim=1)
        msa = torch.cat((msa, seq1hot, state), dim=-1)
        msa = self.norm_node(self.embed_x(msa))
        pair = self.norm_edge1(self.embed_e1(pair))
        
        neighbor = get_seqsep(idx)
        rbf_feat = rbf(torch.cdist(xyz[:,:,1], xyz[:,:,1]))
        pair = torch.cat((pair, rbf_feat, neighbor), dim=-1)
        pair = self.norm_edge2(self.embed_e2(pair))
        
        # define graph
        G = make_graph(xyz[:,:,1,:], pair, idx, top_k=top_k)
        l1_feats = xyz - xyz[:,:,1,:].unsqueeze(2)
        l1_feats = l1_feats.reshape(B*L, -1, 3)
        
        # apply SE(3) Transformer & update coordinates
        shift = self.se3(G, msa.reshape(B*L, -1, 1), l1_feats)

        state = shift['0'].reshape(B, L, -1) # (B, L, C)
        
        offset = shift['1'].reshape(B, L, 2, 3)
        T = offset[:,:,0,:]
        R = offset[:,:,1,:]
        R_angle = torch.norm(R, dim=-1, keepdim=True) # (B, L, 1)
        R_vector = R / (R_angle+eps) # (B, L, 3)
        R_vector = R_vector.unsqueeze(-2) # (B, L, 1, 3)
        #
        v = l1_feats.reshape(B, L, -1, 3)
        R_dot_v = (R_vector * v).sum(dim=-1, keepdim=True) # (B, L, 3, 1)
        R_cross_v = torch.cross(R_vector.expand(-1, -1, 3, -1), v, dim=-1) # (B, L, 3, 3)
        v_perpendicular = v - R_vector*R_dot_v
        u_parallel = R_vector*R_dot_v # (B, L, 3, 3)
        #
        v_new = v_perpendicular*torch.cos(R_angle).unsqueeze(-2) + R_cross_v*torch.sin(R_angle).unsqueeze(-2) + u_parallel # (B, L, 3, 3)
        #
        xyz = v_new + (xyz[:,:,1]+T).unsqueeze(-2)
        return xyz, state

class Str2MSA(nn.Module):
    def __init__(self, d_msa=64, d_state=8, inner_dim=32, r_ff=4,
                 distbin=[8.0, 12.0, 16.0, 20.0], p_drop=0.1):
        super(Str2MSA, self).__init__()
        self.distbin = distbin
        n_att_head = len(distbin)

        self.norm0 = LayerNorm(d_msa)
        self.norm1 = LayerNorm(d_state)
        self.attn = MaskedDirectMultiheadAttention(d_state, d_msa, n_att_head, d_k=inner_dim, dropout=p_drop) 
        self.dropout1 = nn.Dropout(p_drop)

        self.norm2 = LayerNorm(d_msa)
        self.ff = FeedForwardLayer(d_msa, d_msa*r_ff, p_drop=p_drop)
        self.dropout2 = nn.Dropout(p_drop)
        
    def forward(self, msa, xyz, state, idx):
        dist = torch.cdist(xyz[:,:,1], xyz[:,:,1]) # (B, L, L)
        seqsep = idx[:,None,:] - idx[:,:,None]
        seqsep = seqsep.abs()

        mask_s = list()
        for distbin in self.distbin:
            mask_s.append(torch.logical_and(dist > distbin, seqsep > 24))
        mask_s = torch.stack(mask_s, dim=1) # (B, h, L, L)
        
        msa2 = self.norm0(msa)
        state = self.norm1(state)
        msa2 = self.attn(state, state, msa2, mask_s)
        msa = msa + self.dropout1(msa2)

        msa2 = self.norm2(msa)
        msa2 = self.ff(msa2)
        msa = msa + self.dropout2(msa2)

        return msa

class IterBlock(nn.Module):
    def __init__(self, n_layer=1, d_msa=64, d_msa_full=64,
                 d_pair=128, n_head_msa=4, n_head_pair=8, r_ff=4,
                 n_resblock=1, p_drop=0.1, performer_L_opts=None, performer_N_opts=None):
        super(IterBlock, self).__init__()
        
        self.perceiver = Perceiver(n_layer=n_layer, n_att_head=n_head_msa,
                                   n_feat=d_msa, r_ff=r_ff, d_kv=d_msa_full,
                                   performer_opts=performer_N_opts, p_drop=p_drop)
        self.msa2msa = MSA2MSA(n_layer=n_layer, n_att_head=n_head_msa, n_feat=d_msa,
                               r_ff=r_ff, p_drop=p_drop,
                               performer_N_opts=performer_N_opts,
                               performer_L_opts=performer_L_opts)
        self.msa2pair = MSA2Pair(n_feat=d_msa, n_feat_out=d_pair, n_feat_proj=24,
                                 n_resblock=n_resblock, p_drop=p_drop, n_att_head=n_head_msa)
        self.pair2pair = Pair2Pair(n_layer=n_layer, n_att_head=n_head_pair,
                                   n_feat=d_pair, r_ff=r_ff, p_drop=p_drop,
                                   performer_L_opts=performer_L_opts)
        self.pair2msa = Pair2MSA(n_layer=n_layer, n_att_head=4, 
                                 n_feat_in=d_pair, n_feat_out=d_msa, r_ff=r_ff, p_drop=p_drop)

    def forward(self, msa, msa_full, pair, use_transf_checkpoint=False):
        # input:
        #   msa: initial MSA embeddings (N, L, d_msa)
        #   pair: initial residue pair embeddings (L, L, d_pair)
        if use_transf_checkpoint:
            Transformer.USE_CHECKPOINT = True
        
        msa = self.perceiver(msa, msa_full)

        # 1. process MSA features
        msa, att = self.msa2msa(msa)
        
        # 2. update pair features using given MSA
        pair = self.msa2pair(msa, pair, att)

        # 3. process pair features
        pair = self.pair2pair(pair)
        
        # 4. update MSA features using updated pair features
        msa = self.pair2msa(pair, msa)
        
        if use_transf_checkpoint:
            Transformer.USE_CHECKPOINT = False
        
        return msa, pair

class IterBlock_w_Str(nn.Module):
    def __init__(self, n_layer=1, d_msa=64, d_msa_full=64,
                 d_pair=128, n_head_msa=4, n_head_pair=8, r_ff=4,
                 n_resblock=1, p_drop=0.1, performer_L_opts=None, performer_N_opts=None,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}):
        super(IterBlock_w_Str, self).__init__()
        
        self.perceiver = Perceiver(n_layer=n_layer, n_att_head=n_head_msa,
                                   n_feat=d_msa, r_ff=r_ff, d_kv=d_msa_full,
                                   performer_opts=performer_N_opts, p_drop=p_drop)
        self.msa2msa = MSA2MSA(n_layer=n_layer, n_att_head=n_head_msa, n_feat=d_msa,
                               r_ff=r_ff, p_drop=p_drop,
                               performer_N_opts=performer_N_opts,
                               performer_L_opts=performer_L_opts)
        self.msa2pair = MSA2Pair(n_feat=d_msa, n_feat_out=d_pair, n_feat_proj=24,
                                 n_resblock=n_resblock, p_drop=p_drop, n_att_head=n_head_msa)
        self.pair2pair = Pair2Pair(n_layer=n_layer, n_att_head=n_head_pair,
                                   n_feat=d_pair, r_ff=r_ff, p_drop=p_drop,
                                   performer_L_opts=performer_L_opts)
        self.pair2msa = Pair2MSA(n_layer=n_layer, n_att_head=4, 
                                 n_feat_in=d_pair, n_feat_out=d_msa, r_ff=r_ff, p_drop=p_drop)
        self.str2str = Str2Str(d_msa=d_msa, d_pair=d_pair, d_state=SE3_param['l0_out_features'], 
                               SE3_param=SE3_param, p_drop=p_drop)
        self.str2msa = Str2MSA(d_msa=d_msa, r_ff=r_ff, p_drop=p_drop)

    def forward(self, msa, msa_full, pair, xyz, state, seq1hot, idx, top_k=64,
                use_transf_checkpoint=False):
        # input:
        #   msa: initial MSA embeddings (N, L, d_msa)
        #   pair: initial residue pair embeddings (L, L, d_pair)
        #   xyz: previous structure
        #   state: previous state from SE3
        #   seq1hot: 1-hot encoded sequence
        #   idx: residue numbering
        
        if use_transf_checkpoint:
            Transformer.USE_CHECKPOINT = True
            msa = checkpoint.checkpoint(create_custom_forward(self.str2msa), msa, xyz, state, idx)
        else:
            msa = self.str2msa(msa, xyz, state, idx)
            
        # 1. process MSA features
        msa = self.perceiver(msa, msa_full)
        msa, att = self.msa2msa(msa)
        
        # 2. update pair features using given MSA
        pair = self.msa2pair(msa, pair, att)

        # 3. process pair features
        pair = self.pair2pair(pair)
        
        # 4. update MSA features using updated pair features
        msa = self.pair2msa(pair, msa)
        
        if use_transf_checkpoint:
            xyz, state = checkpoint.checkpoint(create_custom_forward(self.str2str, top_k=top_k), msa.float(), pair.float(), xyz.detach().float(), state.float(), seq1hot, idx)
        else:
            xyz, state = self.str2str(msa.float(), pair.float(), xyz.detach().float(), state.float(), seq1hot, idx, top_k=top_k)
            
        if use_transf_checkpoint:
            Transformer.USE_CHECKPOINT = False
        
        return msa, pair, xyz, state

class FinalBlock(nn.Module):
    def __init__(self, n_layer=1, d_msa=64, d_pair=128, n_head_msa=4, n_head_pair=8, r_ff=4,
                 n_resblock=1, p_drop=0.1, performer_L_opts=None, performer_N_opts=None,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}):
        super(FinalBlock, self).__init__()
        
        self.msa2msa = MSA2MSA(n_layer=n_layer, n_att_head=n_head_msa, n_feat=d_msa,
                               r_ff=r_ff, p_drop=p_drop,
                               performer_N_opts=performer_N_opts,
                               performer_L_opts=performer_L_opts)
        self.msa2pair = MSA2Pair(n_feat=d_msa, n_feat_out=d_pair, n_feat_proj=24,
                                 n_resblock=n_resblock, p_drop=p_drop, n_att_head=n_head_msa)
        self.pair2pair = Pair2Pair(n_layer=n_layer, n_att_head=n_head_pair,
                                   n_feat=d_pair, r_ff=r_ff, p_drop=p_drop,
                                   performer_L_opts=performer_L_opts)
        self.pair2msa = Pair2MSA(n_layer=n_layer, n_att_head=4, 
                                 n_feat_in=d_pair, n_feat_out=d_msa, r_ff=r_ff, p_drop=p_drop)
        self.str2msa = Str2MSA(d_msa=d_msa, r_ff=r_ff, p_drop=p_drop)
        self.str2str = Str2Str(d_msa=d_msa, d_pair=d_pair, d_state=SE3_param['l0_out_features'], SE3_param=SE3_param, p_drop=p_drop)
        self.norm_state = LayerNorm(SE3_param['l0_out_features'])
        self.pred_lddt = nn.Linear(SE3_param['l0_out_features'], 1)

    def forward(self, msa, pair, xyz, state, seq1hot, idx, use_transf_checkpoint=False):
        # input:
        #   msa: initial MSA embeddings (N, L, d_msa)
        #   pair: initial residue pair embeddings (L, L, d_pair)
        if use_transf_checkpoint:
            Transformer.USE_CHECKPOINT = True
        
        if use_transf_checkpoint:
            msa = checkpoint.checkpoint(create_custom_forward(self.str2msa), msa, xyz, state, idx)
        else:
            msa = self.str2msa(msa, xyz, state, idx)
            
        # 1. process MSA features
        msa, att = self.msa2msa(msa)
        
        # 2. update pair features using given MSA
        pair = self.msa2pair(msa, pair, att)

        # 3. process pair features
        pair = self.pair2pair(pair)
       
        msa = self.pair2msa(pair, msa)

        if use_transf_checkpoint:
            xyz, state = checkpoint.checkpoint(create_custom_forward(self.str2str, top_k=32), msa.float(), pair.float(), xyz.detach().float(), state.float(), seq1hot, idx)
            Transformer.USE_CHECKPOINT = False
        else:
            xyz, state = self.str2str(msa.float(), pair.float(), xyz.detach().float(), state, seq1hot, idx)
        
        lddt = self.pred_lddt(self.norm_state(state))
        return msa, pair, xyz, lddt.squeeze(-1)

class IterativeFeatureExtractor(nn.Module):
    def __init__(self, n_module=4, n_module_str=4, n_layer=4, d_msa_full=64, d_msa=256, d_pair=128, d_hidden=64,
                 n_head_msa=8, n_head_pair=8, r_ff=4, 
                 n_resblock=1, p_drop=0.1,
                 performer_L_opts=None, performer_N_opts=None,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 device0=0, device1=0):
        super(IterativeFeatureExtractor, self).__init__()
        self.n_module = n_module
        self.n_module_str = n_module_str
        self.device0 = device0
        self.device1 = device1

        self.initial = Pair2Pair(n_layer=n_layer, n_att_head=n_head_pair,
                                 n_feat=d_pair, r_ff=r_ff, p_drop=p_drop,
                                 performer_L_opts=performer_L_opts).to(device0)

        if self.n_module > 0:
            self.iter_block_1 = _get_clones(IterBlock(n_layer=n_layer,
                                                      d_msa_full=d_msa_full,
                                                      d_msa=d_msa, d_pair=d_pair,
                                                      n_head_msa=n_head_msa,
                                                      n_head_pair=n_head_pair,
                                                      r_ff=r_ff,
                                                      n_resblock=n_resblock,
                                                      p_drop=p_drop,
                                                      performer_N_opts=performer_N_opts,
                                                      performer_L_opts=performer_L_opts
                                                      ).to(device0), n_module)
        
        self.init_str = InitStr_Network(node_dim_in=d_msa, node_dim_hidden=d_hidden,
                                        edge_dim_in=d_pair, edge_dim_hidden=d_hidden,
                                        state_dim=SE3_param['l0_out_features'],
                                        nheads=4, nblocks=3, dropout=p_drop).to(device0)
        # make better initial coordinate well matched to current node & edge features
        self.init_str2str = Str2Str(d_msa=d_msa, d_pair=d_pair, d_state=SE3_param['l0_out_features'], 
                                    SE3_param=SE3_param, p_drop=p_drop).to(device1)

        if self.n_module_str > 0:
            self.iter_block_2 = _get_clones(IterBlock_w_Str(n_layer=n_layer, 
                                                      d_msa_full=d_msa_full,
                                                      d_msa=d_msa, d_pair=d_pair,
                                                      n_head_msa=n_head_msa,
                                                      n_head_pair=n_head_pair,
                                                      r_ff=r_ff,
                                                      n_resblock=n_resblock,
                                                      p_drop=p_drop,
                                                      performer_N_opts=performer_N_opts,
                                                      performer_L_opts=performer_L_opts,
                                                      SE3_param=SE3_param
                                                      ).to(device1), n_module_str)
        
        self.norm_state = LayerNorm(SE3_param['l0_out_features']).to(device1)
        self.pred_lddt = nn.Linear(SE3_param['l0_out_features'], 1).to(device1)

    def forward(self, msa, msa_full, pair, seq1hot, idx, use_transf_checkpoint=False):
        # input:
        #   msa: initial MSA embeddings (N, L, d_msa)
        #   pair: initial residue pair embeddings (L, L, d_pair)
        
        pair = self.initial(pair)
        
        if self.n_module > 0:
            for i_m in range(self.n_module):
                torch.cuda.empty_cache()
                #print (i_m)
                msa, pair = self.iter_block_1[i_m](msa, msa_full, pair,\
                            use_transf_checkpoint=use_transf_checkpoint)
                # extract features from MSA & update original pair features
                #if i_m == self.n_module-1:
                #    msa.requires_grad = True
                #    pair.requires_grad = True
                #    msa, pair = self.iter_block_1[i_m](msa, pair,\
                #                use_transf_checkpoint=use_transf_checkpoint)
                #else:
                #    msa, pair = self.iter_block_1[i_m](msa, pair,\
                #                use_transf_checkpoint=False)

        seq1hot = seq1hot.to(self.device1)
        idx = idx.to(self.device1)
        msa = msa.to(self.device1)
        pair = pair.to(self.device1)

        xyz_s = list()
        xyz, state = self.init_str(seq1hot, idx, msa, pair)
        xyz_s.append(xyz)

        if use_transf_checkpoint:
            xyz, state = checkpoint.checkpoint(create_custom_forward(self.init_str2str, top_k=64), msa.float(), pair.float(), xyz.detach().float(), state.float(), seq1hot, idx)
        else:
            xyz, state = self.init_str2str(msa.float(), pair.float(), xyz.detach().float(), state.float(), seq1hot, idx, top_k=64)
        xyz_s.append(xyz)

        if self.n_module_str > 0:
            for i_m in range(self.n_module_str):
                msa, pair, xyz, state = self.iter_block_2[i_m](msa, msa_full, pair, xyz, state, seq1hot, idx,
                                                                top_k=64,
                                                                use_transf_checkpoint=use_transf_checkpoint)
                xyz_s.append(xyz)
        
        lddt = self.pred_lddt(self.norm_state(state)).squeeze(-1)
        
        xyz = torch.stack(xyz_s, dim=0)
        return msa, pair, xyz, lddt
