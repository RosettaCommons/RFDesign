import torch

# Loss functions for the training
# 1. BB rmsd loss
# 2. distance loss (or 6D loss?)
# 3. bond geometry loss
# 4. predicted lddt loss

# Currently, only considers N, CA, C atoms. It should be updated once we're starting to handle full atoms!!

def get_t(N, Ca, C, eps=1e-5):
    #N, Ca, C - [I, B, L, 3]
    #R - [I, B, L, 3, 3], det(R)=1, inv(R) = R.T, R is a rotation matrix
    #t - [I, B, L, L, 3] is the global rotation and translation invariant displacement
    v1 = N-Ca # (I, B, L, 3)
    v2 = C-Ca # (I, B, L, 3)
    e1 = v1/(torch.norm(v1, dim=-1, keepdim=True)+eps) # (I, B, L, 3)
    u2 = v2 - (torch.einsum('iblj,iblj->ibl', v2, e1).unsqueeze(-1)*e1) # (I,B,L,3)
    e2 = u2/(torch.norm(u2, dim=-1, keepdim=True)+eps) # (I, B, L, 3)
    e3 = torch.cross(e1, e2, dim=-1) # (I, B, L, 3)
    R = torch.stack((e1,e2,e3), dim=-2) #[I,B,L,3,3] - rotation matrix
    t = Ca.unsqueeze(-2) - Ca.unsqueeze(-3) # (I,B,L,L,3)
    t = torch.einsum('ibljk, iblmk -> iblmj', R, t) # (I,B,L,L,3)
    return t

def calc_str_loss(pred, true, A=20.0, gamma=0.95):
    '''
    Calculate structural loss described in the AlphaFold2 patent
    Input:
        - pred: predicted coordinates (I, B, L, n_atom, 3)
        - true: true coordinates (B, L, n_atom, 3)
    Output: str loss
    '''
    I = pred.shape[0]
    true = true.unsqueeze(0)
    t_tilde_ij = get_t(true[:,:,:,0], true[:,:,:,1], true[:,:,:,2])
    t_ij = get_t(pred[:,:,:,0], pred[:,:,:,1], pred[:,:,:,2])
    
    difference = torch.norm(t_tilde_ij-t_ij,dim=-1) # (I, B, L, L)
    loss = -(torch.nn.functional.relu(1.0-difference/A)).mean(dim=(1,2,3)) # (I)
    
    # weighting loss
    w_loss = torch.pow(torch.full((I,), gamma, device=pred.device), torch.arange(I, device=pred.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    tot_loss = (w_loss * loss).sum()
    return tot_loss, loss.detach() 


@torch.cuda.amp.autocast(enabled=False)
def calc_crd_rmsd(pred, true, log=False, gamma=0.95):
    '''
    Calculate coordinate RMSD
    Input:
        - pred: predicted coordinates (I, B, L, n_atom, 3)
        - true: true coordinates (B, L, n_atom, 3)
    Output: RMSD after superposition
    '''
    def rmsd(V, W, eps=1e-6):
        L = V.shape[1]
        return torch.sqrt(torch.sum((V-W)*(V-W), dim=(1,2)) / L + eps)
    def centroid(X):
        return X.mean(dim=-2, keepdim=True)

    I, B, L, n_atom = pred.shape[:4]
    
    # center to centroid
    pred = pred - centroid(pred.view(I,B,n_atom*L,3)).view(I,B,1,1,3)
    true = true - centroid(true.view(B,n_atom*L,3)).view(B,1,1,3)

    # reshape true crds to match the shape to pred crds
    true = true.unsqueeze(0).expand(I,-1,-1,-1,-1)
    pred = pred.view(I*B, L*n_atom, 3)
    true = true.view(I*B, L*n_atom, 3)

    # Computation of the covariance matrix
    C = torch.matmul(pred.permute(0,2,1), true)

    # Compute optimal rotation matrix using SVD
    V, S, W = torch.svd(C)

    # get sign to ensure right-handedness
    d = torch.ones([I*B,3,3], device=pred.device)
    d[:,:,-1] = torch.sign(torch.det(V)*torch.det(W)).unsqueeze(1)

    # Rotation matrix U
    U = torch.matmul(d*V, W.permute(0,2,1)) # (IB, 3, 3)

    # Rotate pred
    rP = torch.matmul(pred, U) # (IB, L*3, 3)

    # get RMS
    rms_raw = rmsd(rP, true).reshape(I,B)
    if log:
        rms = torch.log(rms_raw+1.0) # (I, B)
    else:
        rms = rms_raw
    rms = rms.mean(dim=-1) # (I)

    # weighting loss
    w_loss = torch.pow(torch.full((I,), gamma, device=pred.device), torch.arange(I, device=pred.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    tot_rms = (w_loss * rms).sum()
    return tot_rms, rms_raw.mean(dim=-1).detach()

def calc_dist_rmsd(pred, true, max_dist=40.0, max_error=64.0, log=False, gamma=0.95, eps=1e-6):
    '''
    Calculate distance RMSD
    Input:
        - pred: predicted coordinates (I, B, L, n_atom, 3)
        - true: true coordinates (B, L, n_atom, 3)
    Output: RMSD after superposition
    '''
    I, B, L, n_atom = pred.shape[:4]
    pred = pred.reshape(I, B, L*n_atom, 3)
    true = true.reshape(B, L*n_atom, 3)

    D_pred = torch.triu(torch.cdist(pred, pred), diagonal=1) # (I, B, L*n_atom, L*n_atom)
    D_true = torch.triu(torch.cdist(true, true), diagonal=1) # (B, L*n_atom, L*n_atom)
    mask = (D_true < max_dist)
    mask = torch.triu(mask, diagonal=1)
    #
    diff_sq = torch.square(D_pred - D_true[None]) # (I, B, L*n_atom, L*n_atom)
    #diff_sq = torch.clamp(diff_sq, min=0.0, max=max_error)
    diff_sq = (mask[None] * diff_sq).sum(dim=(1,2,3)) / (mask).sum() # (I)
    rms_raw = torch.sqrt(diff_sq + eps) # (I)

    # weighting loss
    w_loss = torch.pow(torch.full((I,), gamma, device=pred.device), torch.arange(I, device=pred.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    tot_rms = (w_loss * rms_raw).sum()
    return tot_rms, rms_raw.detach()

def calc_dist_rmsle(pred, true, gamma=0.95, eps=1e-6):
    '''
    Calculate distance RMSLE
    Input:
        - pred: predicted coordinates (I, B, L, n_atom, 3)
        - true: true coordinates (B, L, n_atom, 3)
    Output: RMSD after superposition
    '''
    I, B, L, n_atom = pred.shape[:4]
    pred = pred.reshape(I, B, L*n_atom, 3)
    true = true.reshape(B, L*n_atom, 3)

    D_pred = torch.log(torch.triu(torch.cdist(pred, pred), diagonal=1) + 1.0) # (I, B, L*n_atom, L*n_atom)
    D_true = torch.log(torch.triu(torch.cdist(true, true), diagonal=1) + 1.0) # (B, L*n_atom, L*n_atom)
    mask = torch.ones_like(D_true[0])
    mask = torch.triu(mask, diagonal=4)
    n_pair = mask.sum()
    mask = mask.unsqueeze(0).unsqueeze(0)

    diff_sq = torch.square(mask*(D_pred-D_true[None])).sum(dim=(-2, -1)) / n_pair
    rms_raw = torch.sqrt(diff_sq + eps) # (I, B)
    #diff_sq = torch.square(D_true[None]-D_pred).sum(dim=(-2, -1)) / n_pair
    #rms_2 = torch.sqrt(diff_sq + eps) # (I, B)
    #rms_raw = rms_1 + rms_2
    rms_raw = rms_raw.mean(dim=-1)
    rms = rms_raw

    # weighting loss
    w_loss = torch.pow(torch.full((I,), gamma, device=pred.device), torch.arange(I, device=pred.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    tot_rms = (w_loss * rms).sum()
    return tot_rms, rms_raw.detach()

def angle(a, b, c, eps=1e-6):
    '''
    Calculate cos/sin angle between ab and cb
    a,b,c have shape of (B, L, 3)
    '''
    B,L = a.shape[:2]

    u1 = a-b
    u2 = c-b

    u1_norm = torch.norm(u1, dim=-1, keepdim=True) + eps
    u2_norm = torch.norm(u2, dim=-1, keepdim=True) + eps

    # normalize u1 & u2 --> make unit vector
    u1 = u1 / u1_norm
    u2 = u2 / u2_norm
    u1 = u1.reshape(B*L, 3)
    u2 = u2.reshape(B*L, 3)

    # sin_theta = norm(a cross b)/(norm(a)*norm(b))
    # cos_theta = norm(a dot b) / (norm(a)*norm(b))
    sin_theta = torch.norm(torch.cross(u1, u2, dim=1), dim=1, keepdim=True).reshape(B, L, 1) # (B,L,1)
    cos_theta = torch.matmul(u1[:,None,:], u2[:,:,None]).reshape(B, L, 1)
    
    return torch.cat([cos_theta, sin_theta], axis=-1) # (B, L, 2)

def length(a, b):
    return torch.norm(a-b, dim=-1)

def torsion(a,b,c,d, eps=1e-6):
    #A function that takes in 4 atom coordinates:
    # a - [B,L,3]
    # b - [B,L,3]
    # c - [B,L,3]
    # d - [B,L,3]
    # and returns cos and sin of the dihedral angle between those 4 points in order a, b, c, d
    # output - [B,L,2]
    u1 = b-a
    u1 = u1 / (torch.norm(u1, dim=-1, keepdim=True) + eps)
    u2 = c-b
    u2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)
    u3 = d-c
    u3 = u3 / (torch.norm(u3, dim=-1, keepdim=True) + eps)
    #
    t1 = torch.cross(u1, u2, dim=-1) #[B, L, 3]
    t2 = torch.cross(u2, u3, dim=-1)
    t1_norm = torch.norm(t1, dim=-1, keepdim=True)
    t2_norm = torch.norm(t2, dim=-1, keepdim=True)
    
    cos_angle = torch.matmul(t1[:,:,None,:], t2[:,:,:,None])[:,:,0]
    sin_angle = torch.norm(u2, dim=-1,keepdim=True)*(torch.matmul(u1[:,:,None,:], t2[:,:,:,None])[:,:,0])
    
    cos_sin = torch.cat([cos_angle, sin_angle], axis=-1)/(t1_norm*t2_norm+eps) #[B,L,2]
    return cos_sin

def calc_BB_bond_geom(pred, true, eps=1e-6):
    '''
    Calculate backbone bond geometry (bond length and angle) and put loss on them
    Input:
     - pred: predicted coords (I, B, L, :, 3), 0; N / 1; CA / 2; C
     - true: True coords (B, L, :, 3)
    Output:
     - bond length loss, bond angle loss
    '''
    I, B, L = pred.shape[:3]
    pred = pred.view(I*B, L, -1, 3)

    # bond length: N-CA, CA-C, C-N
    #blen_NCA_pred = length(pred[:,:,0], pred[:,:,1]).reshape(I, B, L) # (I, B, L)
    #blen_CAC_pred = length(pred[:,:,1], pred[:,:,2]).reshape(I, B, L)
    blen_CN_pred  = length(pred[:,:-1,2], pred[:,1:,0]).reshape(I,B,L-1) # (I, B, L-1)
    
    #blen_NCA_true = length(true[:,:,0], true[:,:,1]) # (B, L)
    #blen_CAC_true = length(true[:,:,1], true[:,:,2])
    blen_CN_true  = length(true[:,:-1,2], true[:,1:,0]) # (B, L-1)
    mask_CN = blen_CN_true < 3.0

    blen_loss = 0.0
    #blen_loss += torch.sqrt(torch.square(blen_NCA_pred - blen_NCA_true[None]) + eps).mean()
    #blen_loss += torch.sqrt(torch.square(blen_CAC_pred - blen_CAC_true[None]) + eps).mean()
    blen_loss += torch.sqrt(torch.square(blen_CN_pred  - blen_CN_true[None])*mask_CN[None] + eps).sum() / (I*mask_CN.sum())

    # bond angle: N-CA-C, CA-C-N, C-N-CA
    #bang_NCAC_pred = angle(pred[:,:,0], pred[:,:,1], pred[:,:,2]).reshape(I,B,L,2)
    bang_CACN_pred = angle(pred[:,:-1,1], pred[:,:-1,2], pred[:,1:,0]).reshape(I,B,L-1,2)
    bang_CNCA_pred = angle(pred[:,:-1,2], pred[:,1:,0], pred[:,1:,1]).reshape(I,B,L-1,2)

    #bang_NCAC_true = angle(true[:,:,0], true[:,:,1], true[:,:,2])
    bang_CACN_true = angle(true[:,:-1,1], true[:,:-1,2], true[:,1:,0])
    bang_CNCA_true = angle(true[:,:-1,2], true[:,1:,0], true[:,1:,1])

    bang_loss = 0.0
    #bang_loss += torch.sqrt(torch.square(bang_NCAC_pred - bang_NCAC_true[None]).sum(dim=-1) + eps).mean()
    bang_loss += torch.sqrt(torch.square(bang_CACN_pred - bang_CACN_true[None]).sum(dim=-1) + eps).mean()
    bang_loss += torch.sqrt(torch.square(bang_CNCA_pred - bang_CNCA_true[None]).sum(dim=-1) + eps).mean()

    return blen_loss, bang_loss

def calc_pseudo_dih(pred, true, eps=1e-6):
    '''
    calculate pseudo CA dihedral angle and put loss on them
    Input:
    - predicted & true CA coordinates (I,B,L,3) / (B, L, 3)
    Output:
    - dihedral angle loss
    '''
    I, B, L = pred.shape[:3]
    pred = pred.reshape(I*B, L, -1)
    true_dih = torsion(true[:,:-3,:],true[:,1:-2,:],true[:,2:-1,:],true[:,3:,:]) # (B, L', 2)
    pred_dih = torsion(pred[:,:-3,:],pred[:,1:-2,:],pred[:,2:-1,:],pred[:,3:,:]) # (I*B, L', 2)
    pred_dih = pred_dih.reshape(I, B, -1, 2)
    dih_loss = torch.sqrt(torch.square(pred_dih - true_dih[None]).sum(dim=-1) + eps).mean()
    return dih_loss


def calc_lddt_loss(pred_ca, true_ca, pred_lddt, idx, eps=1e-6):
    # Input
    # pred_ca: predicted CA coordinates (I, B, L, 3)
    # true_ca: true CA coordinates (B, L, 3)
    # pred_lddt: predicted lddt values (I-1, B, L)

    I, B, L = pred_ca.shape[:3]
    seqsep = torch.abs(idx[:,None,:] - idx[:,:,None]).unsqueeze(0)
    
    pred_dist = torch.cdist(pred_ca, pred_ca) # (I, B, L, L)
    true_dist = torch.cdist(true_ca, true_ca).unsqueeze(0) # (1, B, L, L)

    mask = torch.logical_and(true_dist > 0.0, true_dist < 15.0) # (1, B, L, L)
    #mask = torch.logical_and(mask, seqsep > 11)
    delta = torch.abs(pred_dist-true_dist) # (I, B, L, L)

    true_lddt = torch.zeros((I,B,L), device=pred_ca.device)
    for distbin in [0.5, 1.0, 2.0, 4.0]:
        true_lddt += 0.25*torch.sum((delta<=distbin)*mask, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    #diff = torch.abs(pred_lddt - true_lddt) # (I, B, L)
    diff = torch.square(pred_lddt - true_lddt[-1]) # (I, B, L)
    return diff.mean(), true_lddt.mean(dim=(1,2))
