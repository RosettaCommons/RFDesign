import torch
import torch.nn as nn
from resnet import ResidualNetwork
from Transformer import LayerNorm

class TorsionNetwork(nn.Module):
    def __init__(self, n_feat, p_drop=0.1):
        super(TorsionNetwork, self).__init__()
        self.norm = nn.LayerNorm(n_feat)
        #
        self.proj = nn.Linear(n_feat, 36*2)

    def forward(self, x):
        # input: msa info (B, N, L, C)
        x = self.norm(x[:,0])
        
        logits = self.proj(x).permute(0,2,1)
        logits_phi = logits[:,:36]
        logits_psi = logits[:,36:]
        
        return logits_phi, logits_psi
