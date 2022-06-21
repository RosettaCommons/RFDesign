import sys
import os

import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

from util_module import init_lecun_normal
from opt_einsum import contract as einsum
from constant import INIT_CRDS

def get_xyz(Rin, Tin):
    B, L = Rin.shape[:2]
    # start from zeros
    xyz = INIT_CRDS.to(Rin.device).reshape(1,1,3,3).expand(B,L,-1,-1)

    xyz_rot = einsum('bnij,bnaj->bnai', Rin, xyz)
    return xyz_rot + Tin[:,:,None,:]


class InvariantPointAttention(nn.Module):
    def __init__(self, Nhead=12, c=16,  # channels internally
                 Nq=4, Np=8, 
                 cm=384, # channels per node in
                 cz=64  # channels per edge in
                ):
        super(InvariantPointAttention, self).__init__()

        self.Nhead = Nhead
        self.c = c
        self.Nq = Nq
        self.Np = Np

        # query, key, values from 1D features
        self.linear_q = init_lecun_normal( nn.Linear(cm, Nhead*c, bias=False) )
        self.linear_k = init_lecun_normal( nn.Linear(cm, Nhead*c, bias=False) )
        self.linear_v = init_lecun_normal( nn.Linear(cm, Nhead*c, bias=False) )

        # bias from 2D features
        self.linear_b = init_lecun_normal( nn.Linear(cz, Nhead, bias=False) )

        # structure
        self.gamma = nn.Parameter( np.log(np.exp(1.) - 1.)*torch.ones(Nhead) )
        self.linear_qp = init_lecun_normal( nn.Linear (cm, 3*Nhead*Nq, bias=False) )
        self.linear_kp = init_lecun_normal( nn.Linear (cm, 3*Nhead*Nq, bias=False) )
        self.linear_vp = init_lecun_normal( nn.Linear (cm, 3*Nhead*Np, bias=False) )

        # final
        self.linear_si = nn.Linear (Nhead*(cz+c+4*Np), cm)

        # weights
        self.w_scalar = np.sqrt( 1./(3*c) ) # (w_L * np.sqrt(1/c))
        self.w_point = np.sqrt( 2./(3*9*Nq) ) # (w_L * w_c in AF paper)
        self.w_attn = np.sqrt( 1./3 )# (w_L in AF paper)
        
        self.reset_parameter()

    def reset_parameter(self):
        # initialize finaly layers with zeros
        nn.init.zeros_(self.linear_si.weight)
        nn.init.zeros_(self.linear_si.bias)

    def forward(self, s, z, Rs, Ts, mask=None):
        '''
        Inputs:
        - s: sequence representation (B, L, cm)
        - z: pair representation (B, L, L, cz)
        - Rs: rotation to global framework (B, L, 3, 3)
        - Ts: translation to global framework (B, L, 3)
        - mask: residues to ignore
        Ouputs:
        '''

        B, L = s.shape[:2]

        qi_scalar = self.linear_q(s).reshape(B, L, self.Nhead, self.c) # (B, L, h, c)
        ki_scalar = self.linear_k(s).reshape(B, L, self.Nhead, self.c)
        vi_scalar = self.linear_v(s).reshape(B, L, self.Nhead, self.c)

        qi_local = self.linear_qp(s).reshape(B, L, self.Nhead, self.Nq, 3) # (B, L, h, Nq, 3)
        ki_local = self.linear_kp(s).reshape(B, L, self.Nhead, self.Nq, 3)
        vi_local = self.linear_vp(s).reshape(B, L, self.Nhead, self.Np, 3)
        
        Ts = Ts.view(B, L, 1, 1, 3)
        qi_global = einsum('birq, bihpq -> bihpr', Rs, qi_local)+Ts # (B, L, h, Nq, 3)
        ki_global = einsum('birq, bihpq -> bihpr', Rs, ki_local)+Ts
        vi_global = einsum('birq, bihpq -> bihpr', Rs, vi_local)+Ts # (B, L, h, Np, 3)

        dist2 = torch.square(qi_global[:,:,None,:,:,:] - ki_global[:,None,:,:,:,:]).sum([-1,-2]) # (B, L, L, h)

        attn_qk_point = -0.5 * nn.Softplus()(self.gamma)[None,None,None,:] * self.w_point * dist2 # (B, L, L, h)
        attn_qk_scalar = einsum('bihk, bjhk -> bijh', qi_scalar, ki_scalar*self.w_scalar) # (B, L, L, h)
        attn_logits = attn_qk_scalar + attn_qk_point

        attn2d = self.w_attn*self.linear_b(z) # (B, L, L, h)
        attn_logits = attn_logits + attn2d

        if mask is not None:
            mask_2d = mask[:,:,None] * mask[:,None,:] #[B,L,L]
            attn_logits = attn_logits - 1e5 * (1. - mask_2d[:,:,:,None]) #[B,L,L,h]

        attn = F.softmax(attn_logits, dim=2) #over j

        result_scalar = einsum('bijh, bjhk -> bihk', attn, vi_scalar) #[B,L,h,c]
        result_global = einsum('bijh, bjhpr -> bihpr', attn, vi_global) #[B,L,h,Np,3]
        result_local = einsum('biqr, bihpq -> bihpr', Rs, result_global-Ts) #[B,L,h,Np,3]

        result_local_norm = torch.norm(result_local, dim=-1)

        result_attention_over_2d = einsum('bijh, bijk->bihk', attn, z) #[B, L, h, cz]

        O = torch.cat([
            result_attention_over_2d.reshape(B,L,-1),
            result_scalar.reshape(B,L,-1),
            result_local.reshape(B,L,-1), 
            result_local_norm.reshape(B,L,-1)
        ], axis=-1 )

        si = self.linear_si(O)

        return si

class BackboneUpdate(nn.Module):
    def __init__(self, d_single=384, d_out=6):
        super(BackboneUpdate, self).__init__()
        self.linear = nn.Linear(d_single, d_out)
        self.scale = 10.0 # translation in nm scale

        self.reset_parameter()

    def reset_parameter(self):
        # weight initilize to zeros (initially, rotation & translation = identity)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, s, Rin, Tin):
        B, L = s.shape[:2]

        RT = self.linear(s) # (B, L, C)

        # convert (non-unit) quaternion to rotation matrix
        Qnorm = torch.sqrt(1.0 + torch.sum(RT[:,:,:3]*RT[:,:,:3], dim=-1))
        ai = 1.0/Qnorm
        bi = RT[:,:,0]/Qnorm
        ci = RT[:,:,1]/Qnorm
        di = RT[:,:,2]/Qnorm

        delRi = torch.empty((B, L, 3, 3), device=s.device)
        delRi[:,:,0,0] = ai*ai + bi*bi - ci*ci - di*di
        delRi[:,:,0,1] = 2*bi*ci - 2*ai*di
        delRi[:,:,0,2] = 2*bi*di + 2*ai*ci
        delRi[:,:,1,0] = 2*bi*ci + 2*ai*di
        delRi[:,:,1,1] = ai*ai - bi*bi + ci*ci - di*di
        delRi[:,:,1,2] = 2*ci*di - 2*ai*bi
        delRi[:,:,2,0] = 2*bi*di - 2*ai*ci
        delRi[:,:,2,1] = 2*ci*di + 2*ai*bi
        delRi[:,:,2,2] = ai*ai - bi*bi - ci*ci + di*di

        delTi = self.scale*RT[:,:,3:]
        
        Ri = einsum("bnij,bnjk->bnik", delRi, Rin)
        Ti = einsum("bnij,bnj->bni", delRi, Tin) + delTi
        return Ri, Ti
