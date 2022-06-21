import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from opt_einsum import contract as einsum
import torch.utils.checkpoint as checkpoint
from util_module import *
from Attention_module import *
from InitStrGenerator import InitStr_Network
from SE3_network import SE3TransformerWrapper
from resnet import ResidualNetwork

# Perceiver architecture to update seed MSA w/ extra MSA
class Perceiver(nn.Module):
    def __init__(self, d_msa=256, d_msa_full=64,  n_head=8,
                 d_hidden=32, r_ff=2, p_drop=0.1):
        super(Perceiver, self).__init__()
        self.norm_latent = nn.LayerNorm(d_msa)
        self.norm_full = nn.LayerNorm(d_msa_full)

        # Cross attention 
        self.attn = Attention(d_msa, d_msa_full, n_head, d_hidden, d_msa, p_drop=p_drop)
        self.ff = FeedForwardLayer(d_msa, r_ff, p_drop=p_drop)

    def forward(self, msa_latent, msa_full):
        B, N1, L1 = msa_latent.shape[:3]
        B, N2, L2 = msa_full.shape[:3]
        
        msa_full = self.norm_full(msa_full)
        msa_full = msa_full.permute(0,2,1,3).reshape(B*L2, N2, -1)
        
        msa_latent2 = self.norm_latent(msa_latent)
        msa_latent2 = msa_latent2.permute(0,2,1,3).reshape(B*L1, N1, -1)
        
        msa_latent = msa_latent + self.attn(msa_latent2, msa_full, msa_full).reshape(B, L1, N1, -1).permute(0,2,1,3)
        msa_latent = msa_latent + self.ff(msa_latent)
        
        msa_latent = msa_latent.reshape(B, L1, N1, -1).permute(0,2,1,3).contiguous()
        return msa_latent

# Update MSA information based on given MSA & Pair features
class MSAPair2MSA(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_hidden=32, use_global_attn=False, p_drop=0.15):
        super(MSAPair2MSA, self).__init__()
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.row_attn = MSARowAttentionWithPair(d_msa=d_msa, d_pair=d_pair,
                                                n_head=n_head, d_hidden=d_hidden) 
        if use_global_attn:
            self.col_attn = MSAColGlobalAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        else:
            self.col_attn = MSAColAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        self.ff = FeedForwardLayer(d_msa, 4, p_drop=p_drop)

    def forward(self, msa, pair, use_checkpoint=False):
        if use_checkpoint:
            msa = msa + self.drop_row(checkpoint.checkpoint(create_custom_forward(self.row_attn), msa, pair))
            msa = msa + checkpoint.checkpoint(create_custom_forward(self.col_attn), msa)
            msa = msa + checkpoint.checkpoint(create_custom_forward(self.ff), msa)
        else:
            msa = msa + self.drop_row(self.row_attn(msa, pair))
            msa = msa + self.col_attn(msa)
            msa = msa + self.ff(msa)

        return msa

class MSAPairStr2MSA(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_state=16, d_hidden=32, p_drop=0.15):
        super(MSAPairStr2MSA, self).__init__()
        self.norm_pair = nn.LayerNorm(d_pair)
        self.proj_pair = nn.Linear(d_pair+36, d_pair)
        self.norm_state = nn.LayerNorm(d_state)
        self.proj_state = nn.Linear(d_state, d_msa)
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.row_attn = MSARowAttentionWithPair(d_msa=d_msa, d_pair=d_pair,
                                                n_head=n_head, d_hidden=d_hidden) 
        self.col_attn = MSAColAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        self.ff = FeedForwardLayer(d_msa, 4, p_drop=p_drop)
    
        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distrib
        self.proj_pair = init_lecun_normal(self.proj_pair)
        self.proj_state = init_lecun_normal(self.proj_state)

        # initialize bias to zeros
        nn.init.zeros_(self.proj_pair.bias)
        nn.init.zeros_(self.proj_state.bias)

    def forward(self, msa, pair, xyz, state):
        B, N, L = msa.shape[:3]
        rbf_feat = rbf(torch.cdist(xyz[:,:,1], xyz[:,:,1]))
        pair = self.norm_pair(pair)
        pair = torch.cat((pair, rbf_feat), dim=-1)
        pair = self.proj_pair(pair)
        #
        state = self.norm_state(state)
        state = self.proj_state(state)
        msa = msa + state.unsqueeze(1).expand(-1, N, -1, -1)
        #
        msa = msa + self.drop_row(self.row_attn(msa, pair))
        msa = msa + self.col_attn(msa)
        msa = msa + self.ff(msa)

        return msa

class MSA2Pair(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_hidden=32, p_drop=0.15):
        super(MSA2Pair, self).__init__()
        self.norm = nn.LayerNorm(d_msa)
        self.proj_left = nn.Linear(d_msa, d_hidden)
        self.proj_right = nn.Linear(d_msa, d_hidden)
        self.proj_out = nn.Linear(d_hidden*d_hidden, d_pair)

        self.proj_down = nn.Linear(d_pair*2, d_pair)
        self.update = ResidualNetwork(1, d_pair, d_pair, d_pair, p_drop=p_drop)
        
        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.proj_left = init_lecun_normal(self.proj_left)
        self.proj_right = init_lecun_normal(self.proj_right)
        self.proj_out = init_lecun_normal(self.proj_out)
        nn.init.zeros_(self.proj_left.bias)
        nn.init.zeros_(self.proj_right.bias)
        nn.init.zeros_(self.proj_out.bias)

        # Identity initialization for proj_down
        nn.init.eye_(self.proj_down.weight)
        nn.init.zeros_(self.proj_down.bias)

    def forward(self, msa, pair):
        B, N, L = msa.shape[:3]
        msa = self.norm(msa)
        left = self.proj_left(msa)
        right = self.proj_right(msa)
        right = right / float(N)
        out = einsum('bsli,bsmj->blmij', left, right).reshape(B, L, L, -1)
        out = self.proj_out(out)
        
        pair = torch.cat((pair, out), dim=-1) # (B, L, L, d_pair*2)
        pair = self.proj_down(pair)
        pair = self.update(pair.permute(0,3,1,2).contiguous())
        pair = pair.permute(0,2,3,1).contiguous()
        
        return pair

# Update pair information (from AlphaFold paper)
class Pair2Pair(nn.Module):
    def __init__(self, d_pair=128, n_head=4, d_hidden=32, p_drop=0.15):
        super(Pair2Pair, self).__init__()
        self.drop_raw = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)
        self.tri_attn_start = TriangleAttention(d_pair, n_head=n_head, d_hidden=d_hidden,
                                                p_drop=p_drop)
        self.tri_attn_end = TriangleAttention(d_pair, n_head, d_hidden=d_hidden,
                                              p_drop=p_drop, start_node=False)
        self.tri_mul_out = TriangleMultiplication(d_pair, d_hidden=d_hidden)
        self.tri_mul_in = TriangleMultiplication(d_pair, d_hidden, outgoing=False)
        self.ff = FeedForwardLayer(d_pair, 2)
        
    def forward(self, pair):
        pair = pair + self.drop_raw(self.tri_mul_out(pair))
        pair = pair + self.drop_raw(self.tri_mul_in(pair))
        pair = pair + self.drop_raw(self.tri_attn_start(pair))
        pair = pair + self.drop_col(self.tri_attn_end(pair))
        pair = pair + self.ff(pair)
        return pair

class Str2Str(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_state=16, 
            SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, p_drop=0.1):
        super(Str2Str, self).__init__()
        
        # initial node & pair feature process
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_state = nn.LayerNorm(d_state)
    
        self.embed_x = nn.Linear(d_msa+d_state, SE3_param['l0_in_features'])
        self.embed_e1 = nn.Linear(d_pair, SE3_param['num_edge_features'])
        self.embed_e2 = nn.Linear(SE3_param['num_edge_features']+36+1, SE3_param['num_edge_features'])
        
        self.norm_node = nn.LayerNorm(SE3_param['l0_in_features'])
        self.norm_edge1 = nn.LayerNorm(SE3_param['num_edge_features'])
        self.norm_edge2 = nn.LayerNorm(SE3_param['num_edge_features'])
        
        self.se3 = SE3TransformerWrapper(**SE3_param)
        
        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distribution
        self.embed_x = init_lecun_normal(self.embed_x)
        self.embed_e1 = init_lecun_normal(self.embed_e1)
        self.embed_e2 = init_lecun_normal(self.embed_e2)

        # initialize bias to zeros
        nn.init.zeros_(self.embed_x.bias)
        nn.init.zeros_(self.embed_e1.bias)
        nn.init.zeros_(self.embed_e2.bias)
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, msa, pair, xyz, state, idx, top_k=128, eps=1e-5):
        # process msa & pair features
        B, N, L = msa.shape[:3]
        msa = self.norm_msa(msa[:,0])
        pair = self.norm_pair(pair)
        state = self.norm_state(state)
       
        msa = torch.cat((msa, state), dim=-1)
        msa = self.norm_node(self.embed_x(msa))
        pair = self.norm_edge1(self.embed_e1(pair))
        
        neighbor = get_seqsep(idx)
        rbf_feat = rbf(torch.cdist(xyz[:,:,1], xyz[:,:,1]))
        pair = torch.cat((pair, rbf_feat, neighbor), dim=-1)
        pair = self.norm_edge2(self.embed_e2(pair))
        
        # define graph
        G, edge_feats = make_topk_graph(xyz[:,:,1,:], pair, idx, top_k=top_k)
        l1_feats = xyz - xyz[:,:,1,:].unsqueeze(2)
        l1_feats = l1_feats.reshape(B*L, -1, 3)
        
        # apply SE(3) Transformer & update coordinates
        shift = self.se3(G, msa.reshape(B*L, -1, 1), l1_feats, edge_feats)

        state = shift['0'].reshape(B, L, -1) # (B, L, C)
        
        offset = shift['1'].reshape(B, L, 2, 3) / 100.0
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

class IterBlock_2track_no_perceiver(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head_msa=8, n_head_pair=4,
                 d_hidden_msa=8, d_hidden_pair=32, p_drop=0.15):
        super(IterBlock_2track_no_perceiver, self).__init__()
        
        self.msapair2msa = MSAPair2MSA(d_msa=d_msa, d_pair=d_pair, n_head=n_head_msa, 
                                       d_hidden=d_hidden_msa, use_global_attn=True, p_drop=p_drop)
        self.msa2pair = MSA2Pair(d_msa=d_msa, d_pair=d_pair, d_hidden=d_hidden_pair)
        self.pair2pair = Pair2Pair(d_pair=d_pair, n_head=n_head_pair, d_hidden=d_hidden_pair, p_drop=p_drop)

    def forward(self, msa, pair, use_checkpoint=False):
        # input:
        #   msa: initial MSA embeddings (B, N, L, d_msa)
        #   pair: initial residue pair embeddings (B, L, L, d_pair)
        
        if use_checkpoint:
            # process msa features
            msa = self.msapair2msa(msa, pair, use_checkpoint=use_checkpoint)
            #msa = checkpoint.checkpoint(create_custom_forward(self.msapair2msa), msa, pair)

            # update pair features using given MSA
            pair = checkpoint.checkpoint(create_custom_forward(self.msa2pair), msa, pair)

            # process pair features
            pair = checkpoint.checkpoint(create_custom_forward(self.pair2pair), pair)
        else:
            # process msa features
            msa = self.msapair2msa(msa, pair)

            # update pair features using given MSA
            pair = self.msa2pair(msa, pair)

            # process pair features
            pair = self.pair2pair(pair)
        
        return msa, pair

class IterBlock_2track(nn.Module):
    def __init__(self, d_msa=256, d_msa_full=64, d_pair=128, n_head_msa=8, n_head_pair=4,
                 d_hidden=32, p_drop=0.15):
        super(IterBlock_2track, self).__init__()
        
        #self.perceiver = Perceiver(d_msa=d_msa, d_msa_full=d_msa_full, n_head=n_head_msa,
        #                           d_hidden=d_hidden, r_ff=2, p_drop=p_drop)
        self.msapair2msa = MSAPair2MSA(d_msa=d_msa, d_pair=d_pair, n_head=n_head_msa, 
                                       d_hidden=d_hidden, p_drop=p_drop)
        self.msa2pair = MSA2Pair(d_msa=d_msa, d_pair=d_pair, d_hidden=d_hidden)
        self.pair2pair = Pair2Pair(d_pair=d_pair, n_head=n_head_pair, d_hidden=d_hidden, p_drop=p_drop)

    def forward(self, msa, msa_full, pair, use_checkpoint=False):
        # input:
        #   msa: initial MSA embeddings (B, N, L, d_msa)
        #   msa_full: extra sequences (B, N_extra, L, d_msa_full)
        #   pair: initial residue pair embeddings (B, L, L, d_pair)
        
        if use_checkpoint:
            # update seed msa
            #msa = checkpoint.checkpoint(create_custom_forward(self.perceiver), msa, msa_full)

            # process msa features
            msa = checkpoint.checkpoint(create_custom_forward(self.msapair2msa), msa, pair)

            # update pair features using given MSA
            pair = checkpoint.checkpoint(create_custom_forward(self.msa2pair), msa, pair)

            # process pair features
            pair = checkpoint.checkpoint(create_custom_forward(self.pair2pair), pair)
        else:
            # update seed msa
            #msa = self.perceiver(msa, msa_full)

            # process msa features
            msa = self.msapair2msa(msa, pair)

            # update pair features using given MSA
            pair = self.msa2pair(msa, pair)

            # process pair features
            pair = self.pair2pair(pair)
        
        return msa, pair

class IterBlock_3track(nn.Module):
    def __init__(self, d_msa=256, d_msa_full=64, d_pair=128, n_head_msa=8, n_head_pair=4,
                 d_hidden=32, p_drop=0.15, 
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}):
        super(IterBlock_3track, self).__init__()
        
        #self.perceiver = Perceiver(d_msa=d_msa, d_msa_full=d_msa_full, n_head=n_head_msa,
        #                           d_hidden=d_hidden, r_ff=2, p_drop=p_drop)
        self.msapairstr2msa = MSAPairStr2MSA(d_msa=d_msa, d_pair=d_pair, n_head=n_head_msa, 
                                             d_state=SE3_param['l0_out_features'],
                                             d_hidden=d_hidden, p_drop=p_drop)
        self.msa2pair = MSA2Pair(d_msa=d_msa, d_pair=d_pair, d_hidden=d_hidden)
        self.pair2pair = Pair2Pair(d_pair=128, n_head=n_head_pair, d_hidden=d_hidden, p_drop=p_drop)
        self.str2str = Str2Str(d_msa=d_msa, d_pair=128, d_state=SE3_param['l0_out_features'],
                               SE3_param=SE3_param,
                               p_drop=p_drop)

    def forward(self, msa, msa_full, pair, xyz, state, idx, use_checkpoint=False):
        # input:
        #   msa: initial MSA embeddings (B, N, L, d_msa)
        #   msa_full: extra sequences (B, N_extra, L, d_msa_full)
        #   pair: initial residue pair embeddings (B, L, L, d_pair)
        #   xyz: backbone coordinates (B, L, 3, 3)
        #   state: state featurse (B, L, d_state)
        
        if use_checkpoint:
            # update seed msa
            #msa = checkpoint.checkpoint(create_custom_forward(self.perceiver), msa, msa_full)

            # process msa features
            msa = checkpoint.checkpoint(create_custom_forward(self.msapairstr2msa), msa, pair, xyz, state)

            # update pair features using given MSA
            pair = checkpoint.checkpoint(create_custom_forward(self.msa2pair), msa, pair)

            # process pair features
            pair = checkpoint.checkpoint(create_custom_forward(self.pair2pair), pair)

            # update structure
            xyz, state = checkpoint.checkpoint(create_custom_forward(self.str2str), msa, pair, xyz, state, idx)

        else:
            # update seed msa
            #msa = self.perceiver(msa, msa_full)

            # process msa features
            msa = self.msapairstr2msa(msa, pair, xyz, state)

            # update pair features using given MSA
            pair = self.msa2pair(msa, pair)

            # process pair features
            pair = self.pair2pair(pair)

            # update structure
            xyz, state = self.str2str(msa, pair, xyz, state, idx)
        
        return msa, pair, xyz, state

class IterativeFeatureExtractor(nn.Module):
    def __init__(self, n_module_2track=4, n_module_3track=4, 
                 d_msa=256, d_msa_full=64, d_pair=128, 
                 n_head_msa=8, n_head_pair=4, d_hidden=32,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 p_drop=0.15):
        super(IterativeFeatureExtractor, self).__init__()
        self.n_module_2track = n_module_2track
        self.n_module_3track = n_module_3track

        self.extra_block = nn.ModuleList([IterBlock_2track_no_perceiver(d_msa=d_msa_full,
                                                                        d_pair=d_pair,
                                                                        n_head_msa=n_head_msa,
                                                                        n_head_pair=n_head_pair,
                                                                        d_hidden_msa=8,
                                                                        d_hidden_pair=d_hidden) for i in range(4)])

        if self.n_module_2track > 0:
            self.iter_block_1 = nn.ModuleList([IterBlock_2track(d_msa=d_msa,
                                                             d_msa_full=d_msa_full,
                                                             d_pair=d_pair,
                                                             n_head_msa=n_head_msa,
                                                             n_head_pair=n_head_pair,
                                                             d_hidden=d_hidden,
                                                             p_drop=p_drop) for i in range(n_module_2track)])
        
        self.init_str = InitStr_Network(node_dim_in=d_msa, node_dim_hidden=d_hidden,
                                        edge_dim_in=d_pair, edge_dim_hidden=d_hidden,
                                        state_dim=SE3_param['l0_out_features'],
                                        nheads=4, nblocks=3, dropout=p_drop)

        # make better initial coordinate well matched to current node & edge features
        self.init_str2str = Str2Str(d_msa=d_msa, d_pair=d_pair, d_state=SE3_param['l0_out_features'], 
                                    SE3_param=SE3_param, p_drop=p_drop)

        if self.n_module_3track > 0:
            self.iter_block_2 = nn.ModuleList([IterBlock_3track(d_msa=d_msa,
                                                             d_msa_full=d_msa_full,
                                                             d_pair=d_pair,
                                                             n_head_msa=n_head_msa,
                                                             n_head_pair=n_head_pair,
                                                             d_hidden=d_hidden,
                                                             SE3_param=SE3_param,
                                                             p_drop=p_drop) for i in range(n_module_3track)])
        
        self.norm_state = nn.LayerNorm(SE3_param['l0_out_features'])
        self.pred_lddt = nn.Linear(SE3_param['l0_out_features'], 1)
        
        self.reset_parameter()

    def reset_parameter(self):
        self.pred_lddt = init_lecun_normal(self.pred_lddt)
        nn.init.zeros_(self.pred_lddt.bias)

    def forward(self, msa, msa_full, pair, idx, use_checkpoint=False):
        # input:
        #   msa: initial MSA embeddings (N, L, d_msa)
        #   pair: initial residue pair embeddings (L, L, d_pair)
       
        for i_m in range(4):
            msa_full, pair = self.extra_block[i_m](msa_full, pair, 
                                                   use_checkpoint=use_checkpoint)

        if self.n_module_2track > 0:
            for i_m in range(self.n_module_2track):
                msa, pair = self.iter_block_1[i_m](msa, msa_full, pair,
                                                   use_checkpoint=use_checkpoint)

        xyz_s = list()
        xyz, state = self.init_str(idx, msa, pair)
        xyz_s.append(xyz)
        xyz, state = self.init_str2str(msa.float(), pair.float(), xyz.detach().float(), state.float(), idx, top_k=128)
        #if use_checkpoint:
        #    xyz, state = checkpoint.checkpoint(create_custom_forward(self.init_str2str, top_k=128), msa.float(), pair.float(), xyz.detach().float(), state.float(), idx)
        #else:
        #    xyz, state = self.init_str2str(msa.float(), pair.float(), xyz.detach().float(), state.float(), idx, top_k=128)
        xyz_s.append(xyz)

        if self.n_module_3track > 0:
            for i_m in range(self.n_module_3track):
                msa, pair, xyz, state = self.iter_block_2[i_m](msa, msa_full, pair, xyz, state, idx,
                                                                use_checkpoint=use_checkpoint)
                xyz_s.append(xyz)
        
        lddt = self.pred_lddt(self.norm_state(state)).squeeze(-1)
        
        xyz = torch.stack(xyz_s, dim=0)
        return msa, pair, xyz, lddt
