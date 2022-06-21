import torch
from torch import nn
from torch.nn import functional as F

from constant import INIT_CRDS
from invariant_point_attention import InvariantPointAttention, BackboneUpdate
from util_module import init_lecun_normal
from opt_einsum import contract as einsum

class InitStr_Network(nn.Module):
    def __init__(self, d_str=384, d_pair=128, d_state=32, n_head_str=12, d_hidden_str=16):
        super(InitStr_Network, self).__init__()

        self.norm_str = nn.LayerNorm(d_str)
        self.norm_pair = nn.LayerNorm(d_pair)

        self.proj_init = nn.Linear(d_str, d_str)

        # invariant point attention
        self.ipa = InvariantPointAttention(Nhead=n_head_str, c=d_hidden_str, Nq=4, Np=8, 
                                           cm=d_str, cz=d_pair)
        self.drop_1 = nn.Dropout(0.1)
        self.norm_1 = nn.LayerNorm(d_str)

        # Transition
        self.linear_1 = nn.Linear(d_str, d_str)
        self.linear_2 = nn.Linear(d_str, d_str)
        self.linear_3 = nn.Linear(d_str, d_str)
        self.drop_2 = nn.Dropout(0.1)
        self.norm_2 = nn.LayerNorm(d_str)

        # Update Backbone
        self.bb_update = BackboneUpdate(d_single=d_str)
        
        # make state feature
        self.proj_state = nn.Linear(d_str, d_state)
        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.proj_init = init_lecun_normal(self.proj_init)
        self.linear_3 = init_lecun_normal(self.linear_3)
        self.proj_state = init_lecun_normal(self.proj_state)

        # relu
        nn.init.kaiming_normal_(self.linear_1.weight, nonlinearity='relu') 
        nn.init.kaiming_normal_(self.linear_2.weight, nonlinearity='relu') 

        # biases to zeros
        nn.init.zeros_(self.proj_init.bias)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.zeros_(self.linear_2.bias)
        nn.init.zeros_(self.linear_3.bias)
        nn.init.zeros_(self.proj_state.bias)
        
    def forward(self, str_feat, pair):
        B, L = str_feat.shape[:2]
        str_feat = self.norm_str(str_feat)
        pair = self.norm_pair(pair)
        #
        str_feat = self.proj_init(str_feat)
        #
        # Initial Transformations
        Ri = torch.eye(3, device=pair.device).reshape(1,1,3,3).expand(B,L,-1,-1)
        Ti = torch.zeros((B,L,3), device=pair.device)
        
        str_feat = str_feat + self.ipa(str_feat, pair, Ri.detach(), Ti)
        str_feat = self.norm_1(self.drop_1(str_feat))

        # transition
        str_feat = str_feat + self.linear_3(F.relu(self.linear_2(F.relu(self.linear_1(str_feat)))))
        str_feat = self.norm_2(self.drop_2(str_feat))

        # update backbone
        Ri, Ti = self.bb_update(str_feat, Ri.detach(), Ti)
        xyz = INIT_CRDS.to(str_feat.device).reshape(1,1,3,3).expand(B, L, -1, -1)
        xyz = einsum('bnij,bnaj->bnai', Ri, xyz) + Ti.unsqueeze(-2)
        
        str_feat = self.proj_state(str_feat)
        return xyz, str_feat
