import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Transformer import EncoderLayer, AxialEncoderLayer, Encoder, LayerNorm
from resnet import ResidualNetwork
# Initial embeddings for target sequence, msa, template info
# positional encoding
#   option 1: using sin/cos --> using this for now 
#   option 2: learn positional embedding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, p_drop=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.drop = nn.Dropout(p_drop)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # (1, max_len, d_model)
    def forward(self, x, idx_s):
        pe = list()
        for idx in idx_s:
            pe.append(self.pe[:,idx,:])
        pe = torch.stack(pe)
        x = x + torch.autograd.Variable(pe, requires_grad=False)
        return self.drop(x)

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, p_drop=0.1):
        super(PositionalEncoding2D, self).__init__()
        self.drop = nn.Dropout(p_drop)
        #
        d_model_half = d_model // 2
        div_term = torch.exp(torch.arange(0., d_model_half, 2) *
                             -(math.log(10000.0) / d_model_half))
        self.register_buffer('div_term', div_term)
    
    def forward(self, x, idx_s):
        B, L, _, K = x.shape
        K_half = K//2
        pe = torch.zeros_like(x)
        i_batch = -1
        for idx in idx_s:
            i_batch += 1
            sin_inp = idx.unsqueeze(1) * self.div_term
            emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1) # (L, K//2)
            pe[i_batch,:,:,:K_half] = emb.unsqueeze(1)
            pe[i_batch,:,:,K_half:] = emb.unsqueeze(0)
        x = x + torch.autograd.Variable(pe, requires_grad=False)
        return self.drop(x)

class QueryEncoding(nn.Module):
    def __init__(self, d_model):
        super(QueryEncoding, self).__init__()
        self.pe = nn.Embedding(2, d_model) # (0 for query, 1 for others)
    
    def forward(self, x):
        B, N, L, K = x.shape
        idx = torch.ones((B, N, L), device=x.device).long()
        idx[:,0,:] = 0 # first sequence is the query
        x = x + self.pe(idx)
        return x 

class MSA_emb(nn.Module):
    def __init__(self, d_model=64, d_msa=22, p_drop=0.1, max_len=5000, incl_query=True):
        super(MSA_emb, self).__init__()
        self.incl_query = incl_query
        self.emb = nn.Embedding(d_msa, d_model)
        self.pos = PositionalEncoding(d_model, p_drop=p_drop, max_len=max_len)
        if self.incl_query:
            self.pos_q = QueryEncoding(d_model)
    def forward(self, msa, idx, msa_one_hot=None):
        B, N, L = msa.shape
        if msa_one_hot is not None:
            out = msa_one_hot @ self.emb.weight
        else:
            out = self.emb(msa) # (B, N, L, K//2)
        out = self.pos(out, idx) # add positional encoding
        if self.incl_query:
            return self.pos_q(out) # add query encoding
        else:
            return out

# pixel-wise attention based embedding (from trRosetta-tbm)
class Templ_emb(nn.Module):
    def __init__(self, d_t1d=1, d_t2d=46, d_templ=64, n_att_head=4, r_ff=4,
                 performer_opts=None, p_drop=0.1, max_len=5000):
        super(Templ_emb, self).__init__()
        self.proj = nn.Linear(d_t1d*2+d_t2d, d_templ)
        self.to_sgnl = nn.Conv2d(d_templ, d_templ, 3, padding=1)
        self.to_attn = nn.Conv2d(d_templ, 1, 3, padding=1)
    
    def forward(self, t1d, t2d, idx):
        # Input
        #   - t1d: 1D template info (B, T, L, 2)
        #   - t2d: 2D template info (B, T, L, L, 10)
        B, T, L, _ = t1d.shape
        left = t1d.unsqueeze(3).expand(-1,-1,-1,L,-1)
        right = t1d.unsqueeze(2).expand(-1,-1,L,-1,-1)
        #
        feat = torch.cat((t2d, left, right), -1)
        feat = self.proj(feat).reshape(B*T, L, L, -1).permute(0,3,1,2)
        #
        sgnl = self.to_sgnl(feat).reshape(B, T, -1, L, L) # (B*T, d_templ, L, L)
        attn = self.to_attn(feat).reshape(B, T, -1, L, L) # (B*T, 1, L, L)
        attn = F.softmax(attn, dim=1) # (B, T, 1, L, L)
        feat = attn*sgnl # (B, T, d_templ, L, L)
        feat = feat.sum(dim=1).permute(0,2,3,1)
        return feat.reshape(B, L, L, -1)

class Pair_emb_w_templ(nn.Module):
    def __init__(self, d_model=128, d_seq=21, d_templ=64, p_drop=0.1):
        super(Pair_emb_w_templ, self).__init__()
        self.d_model = d_model
        self.d_emb = d_model // 2
        self.emb = nn.Embedding(d_seq, self.d_emb)
        self.norm_templ = LayerNorm(d_templ)
        self.projection = nn.Linear(d_model + d_templ, d_model)
        self.pos = PositionalEncoding2D(d_model, p_drop=p_drop)

    def forward(self, seq, idx, templ, seq1hot=None):
        # input:
        #   seq: target sequence (B, L, 20)
        B = seq.shape[0]
        L = seq.shape[1]
        #
        # get initial sequence pair features
        if seq1hot != None:
            seq = seq1hot @ self.emb.weight
        else:
            seq = self.emb(seq) # (B, L, d_model//2)

        left  = seq.unsqueeze(2).expand(-1,-1,L,-1)
        right = seq.unsqueeze(1).expand(-1,L,-1,-1)
        #
        templ = self.norm_templ(templ)
        pair = torch.cat((left, right, templ), dim=-1)
        pair = self.projection(pair) # (B, L, L, d_model)
        
        return self.pos(pair, idx)

class Pair_emb_wo_templ(nn.Module):
    #TODO: embedding without template info
    def __init__(self, d_model=128, d_seq=21, p_drop=0.1):
        super(Pair_emb_wo_templ, self).__init__()
        self.d_model = d_model
        self.d_emb = d_model // 2
        self.emb = nn.Embedding(d_seq, self.d_emb)
        self.projection = nn.Linear(d_model, d_model)
        self.pos = PositionalEncoding2D(d_model, p_drop=p_drop)
    def forward(self, seq, idx, seq1hot=None):
        # input:
        #   seq: target sequence (B, L, 20)
        B = seq.shape[0]
        L = seq.shape[1]

        if seq1hot != None:
            seq = seq1hot @ self.emb.weight
        else:
            seq = self.emb(seq) # (B, L, d_model//2)

        left  = seq.unsqueeze(2).expand(-1,-1,L,-1)
        right = seq.unsqueeze(1).expand(-1,L,-1,-1)
        #
        pair = torch.cat((left, right), dim=-1)
        pair = self.projection(pair)
        return self.pos(pair, idx)

