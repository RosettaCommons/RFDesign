import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import LayerNorm, SequenceWeight
from constant import *

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv

def get_seqsep(idx):
    '''
    Input:
        - idx: residue indices of given sequence (B,L)
    Output:
        - seqsep: sequence separation feature with sign (B, L, L, 1)
                  Sergey found that having sign in seqsep features helps a little
    '''
    seqsep = idx[:,None,:] - idx[:,:,None]
    sign = torch.sign(seqsep)
    neigh = torch.abs(seqsep)
    neigh[neigh > 1] = 0.0 # if bonded -- 1.0 / else 0.0
    neigh = sign * neigh
    return neigh.unsqueeze(-1)

def make_graph(node, idx, emb):
    ''' create torch_geometric graph from Trunk outputs '''
    device = emb.device
    B, L = emb.shape[:2]

    # |i-j| <= kmin (connect sequentially adjacent residues)
    sep = idx[:,None,:] - idx[:,:,None]
    sep = sep.abs()
    b, i, j = torch.where(sep > 0)
    
    src = b*L+i
    tgt = b*L+j

    x = node.reshape(B*L, -1)

    G = Data(x=x, 
             edge_index=torch.stack([src,tgt]),
             edge_attr=emb[b,i,j])

    return G

class UniMPBlock(nn.Module):
    '''https://arxiv.org/pdf/2009.03509.pdf'''
    def __init__(self, 
                 node_dim=64,
                 edge_dim=64,
                 heads=4, 
                 dropout=0.15):
        super(UniMPBlock, self).__init__()
        
        self.TConv = TransformerConv(node_dim, node_dim, heads, dropout=dropout, edge_dim=edge_dim)
        self.LNorm = LayerNorm(node_dim*heads)
        self.Linear = nn.Linear(node_dim*heads, node_dim)
        self.Activ = nn.ELU(inplace=True)

    #@torch.cuda.amp.autocast(enabled=True)
    def forward(self, G):
        xin, e_idx, e_attr = G.x, G.edge_index, G.edge_attr
        x = self.TConv(xin, e_idx, e_attr)
        x = self.LNorm(x)
        x = self.Linear(x)
        out = self.Activ(x+xin)
        return Data(x=out, edge_index=e_idx, edge_attr=e_attr)


class InitStr_Network(nn.Module):
    def __init__(self, 
                 node_dim_in=64, 
                 node_dim_hidden=64,
                 edge_dim_in=128, 
                 edge_dim_hidden=64, 
                 state_dim=8,
                 nheads=4, 
                 nblocks=3, 
                 dropout=0.1):
        super(InitStr_Network, self).__init__()

        # embedding layers for node and edge features
        self.norm_node = LayerNorm(node_dim_in)
        self.norm_edge = LayerNorm(edge_dim_in)
        self.encoder_seq = SequenceWeight(node_dim_in, 1, dropout=dropout)

        self.embed_x = nn.Sequential(nn.Linear(node_dim_in+21, node_dim_hidden), LayerNorm(node_dim_hidden))
        self.embed_e = nn.Sequential(nn.Linear(edge_dim_in+1, edge_dim_hidden), LayerNorm(edge_dim_hidden))
        
        # graph transformer
        blocks = [UniMPBlock(node_dim_hidden,edge_dim_hidden,nheads,dropout) for _ in range(nblocks)]
        self.transformer = nn.Sequential(*blocks)
        
        # outputs
        self.get_l1 = nn.Linear(node_dim_hidden,6) # predict CA displacement vector, rotation axis/angle
        self.norm_state = LayerNorm(node_dim_hidden)
        self.get_state = nn.Linear(node_dim_hidden, state_dim)
    
    def forward(self, seq1hot, idx, msa, pair, eps=1e-5):
        B, N, L = msa.shape[:3]
        device = msa.device

        msa = self.norm_node(msa)
        pair = self.norm_edge(pair)
        
        w_seq = self.encoder_seq(msa).reshape(B, L, 1, N).permute(0,3,1,2)
        msa = w_seq*msa
        msa = msa.sum(dim=1)
        node = torch.cat((msa, seq1hot), dim=-1)
        node = self.embed_x(node)

        seqsep = get_seqsep(idx) 
        pair = torch.cat((pair, seqsep), dim=-1)
        pair = self.embed_e(pair)
        
        G = make_graph(node, idx, pair)
        Gout = self.transformer(G)
        
        l1 = self.get_l1(Gout.x).reshape(B, L, 2, 3) # first: CA coords / second: rotation axis/angle
        T = l1[:,:,0,:] # (B, L, 3) translation
        R = l1[:,:,1,:]
        R_angle = torch.norm(R, dim=-1, keepdim=True) # (B, L, 1)
        R_vector = R / (R_angle+eps) # (B, L, 3)
        R_vector = R_vector.unsqueeze(-2) # (B, L, 1, 3)
        #
        v = INIT_CRDS.to(device).reshape(1,1,3,3).expand(B, L, -1, -1)
        #
        R_dot_v = (R_vector * v).sum(dim=-1, keepdim=True) # (B, L, 3, 1)
        R_cross_v = torch.cross(R_vector.expand(-1, -1, 3, -1), v, dim=-1) # (B, L, 3, 3)
        v_perpendicular = v - R_vector*R_dot_v
        u_parallel = R_vector*R_dot_v # (B, L, 3, 3)
        #
        xyz = v_perpendicular*torch.cos(R_angle).unsqueeze(-2) + R_cross_v*torch.sin(R_angle).unsqueeze(-2) + u_parallel # (B, L, 3, 3)
        xyz = xyz + T.unsqueeze(-2)

        state = self.norm_state(Gout.x)
        state = self.get_state(state)
        return xyz.reshape(B, L, 3, 3) , state.reshape(B, L, -1)
