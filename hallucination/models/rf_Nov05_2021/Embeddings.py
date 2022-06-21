import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract as einsum
import torch.utils.checkpoint as checkpoint
from util_module import Dropout, get_clones, create_custom_forward, rbf, init_lecun_normal
from Attention_module import Attention, TriangleMultiplication, TriangleAttention, FeedForwardLayer

# Module contains classes and functions to generate initial embeddings

class PositionalEncoding2D(nn.Module):
    # Add relative positional encoding to pair features
    def __init__(self, d_model, minpos=-32, maxpos=32, p_drop=0.1):
        super(PositionalEncoding2D, self).__init__()
        self.minpos = minpos
        self.maxpos = maxpos
        self.nbin = abs(minpos)+maxpos+1
        self.emb = nn.Embedding(self.nbin, d_model)
        self.drop = nn.Dropout(p_drop)
    
    def forward(self, x, idx):
        bins = torch.arange(self.minpos, self.maxpos, device=x.device)
        seqsep = idx[:,None,:] - idx[:,:,None] # (B, L, L)
        #
        ib = torch.bucketize(seqsep, bins).long() # (B, L, L)
        emb = self.emb(ib) #(B, L, L, d_model)
        x = x + emb # add relative positional encoding
        return self.drop(x)

class MSA_emb(nn.Module):
    # Get initial seed MSA embedding
    def __init__(self, d_msa=256, d_pair=128, d_init=22+22+2,
                 minpos=-32, maxpos=32, p_drop=0.1):
        super(MSA_emb, self).__init__()
        self.emb = nn.Linear(d_init, d_msa) # embedding for general MSA
        self.emb_q = nn.Embedding(22, d_msa) # embedding for query sequence -- used for MSA embedding
        self.emb_left = nn.Embedding(22, d_pair) # embedding for query sequence -- used for pair embedding
        self.emb_right = nn.Embedding(22, d_pair) # embedding for query sequence -- used for pair embedding
        self.drop = nn.Dropout(p_drop)
        self.pos = PositionalEncoding2D(d_pair, minpos=minpos, maxpos=maxpos, p_drop=p_drop)
        
        self.reset_parameter()
    
    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        self.emb_q = init_lecun_normal(self.emb_q)
        self.emb_left = init_lecun_normal(self.emb_left)
        self.emb_right = init_lecun_normal(self.emb_right)

        nn.init.zeros_(self.emb.bias)

    def forward(self, msa, seq, idx, seq1hot=None):
        # Inputs:
        #   - msa: Input MSA (B, N, L, d_init)
        #   - seq: Input Sequence (B, L)
        #   - idx: Residue index
        # Outputs:
        #   - msa: Initial MSA embedding (B, N, L, d_msa)
        #   - pair: Initial Pair embedding (B, L, L, d_pair)

        N = msa.shape[1] # number of sequenes in MSA
        
        # msa embedding
        if seq1hot is not None:
            tmp = (seq1hot @ self.emb_q.weight).unsqueeze(1)
        else:
            tmp = self.emb_q(seq).unsqueeze(1) # (B, 1, L, d_model) -- query embedding
        msa = self.emb(msa) # (B, N, L, d_model) # MSA embedding
        msa = msa + tmp.expand(-1, N, -1, -1) # adding query embedding to MSA
        msa = self.drop(msa)

        # pair embedding 
        if seq1hot is not None:
            left = (seq1hot @ self.emb_left.weight)[:,None] # (B, 1, L, d_pair)
            right = (seq1hot @ self.emb_right.weight)[:,:,None] # (B, L, 1, d_pair)
        else:
            left = self.emb_left(seq)[:,None] # (B, 1, L, d_pair)
            right = self.emb_right(seq)[:,:,None] # (B, L, 1, d_pair)
        pair = left + right # (B, L, L, d_pair)
        pair = self.pos(pair, idx) # add relative position

        return self.drop(msa), pair

class Extra_emb(nn.Module):
    # Get initial seed MSA embedding
    def __init__(self, d_msa=256, d_init=22+22+2, p_drop=0.1):
        super(Extra_emb, self).__init__()
        self.emb = nn.Linear(d_init, d_msa) # embedding for general MSA
        self.emb_q = nn.Embedding(22, d_msa) # embedding for query sequence
        self.drop = nn.Dropout(p_drop)
        
        self.reset_parameter()
    
    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        nn.init.zeros_(self.emb.bias)

    def forward(self, msa, seq, idx, seq1hot=None):
        # Inputs:
        #   - msa: Input MSA (B, N, L, d_init)
        #   - seq: Input Sequence (B, L)
        #   - idx: Residue index
        # Outputs:
        #   - msa: Initial MSA embedding (B, N, L, d_msa)
        N = msa.shape[1] # number of sequenes in MSA
        msa = self.emb(msa) # (B, N, L, d_model) # MSA embedding
        if seq1hot is not None:
            seq = (seq1hot @ self.emb_q.weight).unsqueeze(1) # (B, 1, L, d_model) -- query embedding
        else:
            seq = self.emb_q(seq).unsqueeze(1) # (B, 1, L, d_model) -- query embedding
        msa = msa + seq.expand(-1, N, -1, -1) # adding query embedding to MSA
        return self.drop(msa)

class TemplateProcessor(nn.Module):
    def __init__(self, d_templ=64, n_head=4, d_hidden=64, p_drop=0.25):
        super(TemplateProcessor, self).__init__()
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)
        self.tri_attn_start = TriangleAttention(d_templ, n_head=n_head, d_hidden=d_hidden,
                                                p_drop=p_drop)
        self.tri_attn_end = TriangleAttention(d_templ, n_head, d_hidden=d_hidden,
                                              p_drop=p_drop, start_node=False)
        self.tri_mul_out = TriangleMultiplication(d_templ, d_hidden=d_hidden)
        self.tri_mul_in = TriangleMultiplication(d_templ, d_hidden, outgoing=False)
        self.ff = FeedForwardLayer(d_templ, 2)
   
    def forward(self, templ):
        templ = templ + self.drop_row(self.tri_attn_start(templ))
        templ = templ + self.drop_col(self.tri_attn_end(templ))
        templ = templ + self.drop_row(self.tri_mul_out(templ))
        templ = templ + self.drop_row(self.tri_mul_in(templ))
        templ = templ + self.ff(templ)
        return templ

class TemplatePairStack(nn.Module):
    def __init__(self, n_block=2, d_templ=64, n_head=4, d_hidden=64, p_drop=0.25):
        super(TemplatePairStack, self).__init__()
        self.n_block = n_block
        proc_s = [TemplateProcessor(d_templ=d_templ, n_head=n_head, d_hidden=d_hidden, p_drop=p_drop) for i in range(n_block)]
        self.block = nn.ModuleList(proc_s)
        self.norm = nn.LayerNorm(d_templ)
    def forward(self, templ, use_checkpoint=False):
        B, T, L = templ.shape[:3]
        templ = templ.reshape(B*T, L, L, -1)
        for i_block in range(self.n_block):
            if use_checkpoint:
                templ = checkpoint.checkpoint(create_custom_forward(self.block[i_block]), templ)
            else:
                templ = self.block[i_block](templ)
        return self.norm(templ).reshape(B, T, L, L, -1)

class Templ_emb(nn.Module):
    # Get template embedding
    # Features are
    #   - 37 distogram bins + 6 orientations (43)
    #   - Mask (missing/unaligned) (1)
    #   - tiled AA sequence (20 standard aa + gap)
    def __init__(self, d_t1d=21+1, d_t2d=43+1, d_pair=128, n_block=2, d_templ=64,
                 n_head=4, d_hidden=64, p_drop=0.25):
        super(Templ_emb, self).__init__()
        self.emb = nn.Linear(d_t1d*2+d_t2d, d_templ)
        #
        self.templ_stack = TemplatePairStack(n_block=n_block, d_templ=d_templ, n_head=n_head,
                                             d_hidden=d_hidden, p_drop=p_drop)
        
        # attention between templ & initial pair
        self.attn = Attention(d_pair, d_templ, n_head, d_hidden, d_pair, p_drop=p_drop)
        self.reset_parameter()
    
    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        nn.init.zeros_(self.emb.bias)
    
    def forward(self, t1d, t2d, pair, use_checkpoint=False):
        # Input
        #   - t1d: 1D template info (B, T, L, 2)
        #   - t2d: 2D template info (B, T, L, L, 10)
        B, T, L, _ = t1d.shape
        left = t1d.unsqueeze(3).expand(-1,-1,-1,L,-1)
        right = t1d.unsqueeze(2).expand(-1,-1,L,-1,-1)
        #
        templ = torch.cat((t2d, left, right), -1) # (B, T, L, L, n_templ)
        templ = self.emb(templ) # Template templures (B, T, L, L, d_templ)
        templ = self.templ_stack(templ, use_checkpoint=use_checkpoint) # template templure
        
        # mixing with template information (Template pointwise attention)
        pair = pair.reshape(B*L*L, 1, -1)
        templ = templ.permute(0, 2, 3, 1, 4).reshape(B*L*L, T, -1)
        if use_checkpoint:
            out = checkpoint.checkpoint(create_custom_forward(self.attn), pair, templ, templ)
            out = out.reshape(B, L, L, -1)
        else:
            out = self.attn(pair, templ, templ).reshape(B, L, L, -1)
        #
        pair = pair.reshape(B, L, L, -1)
        pair = pair + out
        #
        return pair

class Recycling(nn.Module):
    def __init__(self, d_msa=256, d_pair=128):
        super(Recycling, self).__init__()
        self.proj_dist = nn.Linear(36, d_pair)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_msa = nn.LayerNorm(d_msa)
        
        self.reset_parameter()
    
    def reset_parameter(self):
        self.proj_dist = init_lecun_normal(self.proj_dist)
        nn.init.zeros_(self.proj_dist.bias)

    def forward(self, msa, pair, xyz):
        # three anchor atoms
        N  = xyz[:,:,0]
        Ca = xyz[:,:,1]
        C  = xyz[:,:,2]

        # recreate Cb given N,Ca,C
        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca    
        
        dist = rbf(torch.cdist(Cb, Cb))
        dist = self.proj_dist(dist)
        pair = dist + self.norm_pair(pair)
        msa = self.norm_msa(msa)
        return msa, pair

