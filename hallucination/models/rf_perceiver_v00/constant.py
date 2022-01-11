import numpy as np
import torch

NCa_bond = torch.tensor(1.453).float()
CaC_bond = torch.tensor(1.526).float()
NCaC_angle = np.deg2rad(109.5)
cos_NCaC = np.cos(NCaC_angle)
sin_NCaC = np.sin(NCaC_angle)
#
init_N = torch.tensor([-NCa_bond.item(), 0.0, 0.0]).float()
init_C = torch.tensor([-CaC_bond.item()*cos_NCaC, CaC_bond.item()*sin_NCaC, 0.0]).float()
init_CA = torch.zeros_like(init_N)
INIT_CRDS = torch.stack((init_N, init_CA, init_C), dim=0) # (3,3) 
