import torch
import torch.nn as nn
from Transformer import LayerNorm

class DistanceNetwork(nn.Module):
    def __init__(self, n_feat, p_drop=0.1):
        super(DistanceNetwork, self).__init__()
        self.norm = nn.LayerNorm(n_feat)
        #
        self.proj_symm = nn.Linear(n_feat, 37*2)
        self.proj_asymm = nn.Linear(n_feat, 37+19)

    def forward(self, x):
        # input: pair info (B, L, L, C)
        x = self.norm(x)

        # predict theta, phi (non-symmetric)
        logits_asymm = self.proj_asymm(x)
        logits_theta = logits_asymm[:,:,:,:37].permute(0,3,1,2)
        logits_phi = logits_asymm[:,:,:,37:].permute(0,3,1,2)

        # predict dist, omega
        x = 0.5 * (x + x.permute(0,2,1,3))
        logits_symm = self.proj_symm(x)
        logits_dist = logits_symm[:,:,:,:37].permute(0,3,1,2)
        logits_omega = logits_symm[:,:,:,37:].permute(0,3,1,2)

        return logits_dist, logits_omega, logits_theta, logits_phi

class TorsionNetwork(nn.Module):
    def __init__(self, n_feat, p_drop=0.1):
        super(TorsionNetwork, self).__init__()
        self.norm = nn.LayerNorm(n_feat)
        #
        self.proj = nn.Linear(n_feat, 36*2)

    def forward(self, x):
        # input: msa info (B, N, L, C)
        x = self.norm(x)
        x = x.mean(dim=1) # (B, L, C)
        
        logits = self.proj(x).permute(0,2,1)
        logits_phi = logits[:,:36]
        logits_psi = logits[:,36:]
        
        return logits_phi, logits_psi

class MaskedTokenNetwork(nn.Module):
    def __init__(self, n_feat, p_drop=0.1):
        super(MaskedTokenNetwork, self).__init__()
        self.norm = nn.LayerNorm(n_feat)
        self.proj = nn.Linear(n_feat, 21)

    def forward(self, x):
        B, N, L = x.shape[:3]
        x = self.norm(x) # (B, N, L, C)
        logits = self.proj(x).permute(0,3,1,2).reshape(B, -1, N*L)

        return logits

