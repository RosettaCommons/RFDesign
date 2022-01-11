import numpy as np
import torch

def lDDT(ca0,ca,s=0.001):
    '''smooth lDDT:
    s=0.35  - good for training (smooth)
    s=0.001 - (or smaller) good for testing
    '''
    L = ca0.shape[0]
    d0 = torch.cdist(ca0,ca0)
    d0 = d0 + 999.9*torch.eye(L,device=ca0.device) # exclude diagonal
    i,j = torch.where(d0<15.0)
    d = torch.cdist(ca,ca)
    dd = torch.abs(d0[i,j]-d[i,j])+1e-3
    def f(x,m,s):
        return 0.5*torch.erf((torch.log(dd)-np.log(m))/(s*2**0.5))+0.5
    lddt = torch.stack([f(dd,m,s) for m in [0.5,1.0,2.0,4.0]],dim=-1).mean()
    return 1.0-lddt


def RMSD(P, Q):
    '''Kabsch algorthm'''
    def rmsd(V, W):
        return torch.sqrt( torch.sum( (V-W)*(V-W) ) / len(V) )
    def centroid(X):
        return X.mean(axis=0)

    cP = centroid(P)
    cQ = centroid(Q)
    P = P - cP
    Q = Q - cQ

    # Computation of the covariance matrix
    C = torch.mm(P.T, Q)

    # Computate optimal rotation matrix using SVD
    V, S, W = torch.svd(C)

    # get sign( det(V)*det(W) ) to ensure right-handedness
    d = torch.ones([3,3],device=P.device)
    d[:,-1] = torch.sign(torch.det(V) * torch.det(W))

    # Rotation matrix U
    U = torch.mm(d*V, W.T)

    # Rotate P
    rP = torch.mm(P, U)

    # get RMS
    rms = rmsd(rP, Q)

    return rms #, rP


def KL(P, Q, eps=1e-6):
    '''KL-divergence between two sets of 6D coords'''
    kl = [(Pi*torch.log((Pi+eps)/(Qi+eps))).sum(0).mean() for Pi,Qi in zip(P,Q)]
    kl = torch.stack(kl).mean()
    return kl
