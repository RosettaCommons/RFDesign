import torch
import torch.nn.functional as F
import numpy as np
import sys, os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir+'/util/')
import geometry

class AffineTransform(object):
    def __init__(self, xyz1, xyz2):
        '''
        Calculates the affine transform to superimpose xyz1 (mobile) onto xyz2 (stationary)
        '''
        assert xyz1.shape == xyz2.shape, 'The input coordinates must have the same shape'
        self.xyz1 = xyz1  # (B, n, 3)
        self.xyz2 = xyz2
        self.B = xyz1.shape[0]
        
    # properties
    @property
    def centroid1(self):
        return self.xyz1.mean(1)
    
    @property
    def centroid2(self):
        return self.xyz2.mean(1)
    
    @property
    def U(self):
        '''Rotation matrix'''
        # center
        xyz1 = self.xyz1 - self.centroid1
        xyz2 = self.xyz2 - self.centroid2

        # Computation of the covariance matrix
        C = torch.matmul(xyz1.permute(0,2,1), xyz2)

        # Compute optimal rotation matrix using SVD
        try:
            V, S, W = torch.svd(C.to(torch.float32))
        except: #incase ill conditioned doesnt work if full of nans
            V, S, W = torch.svd(C.to(torch.float32)+1e-2*C.mean()*torch.rand(C.shape, device=C.device))


        # get sign to ensure right-handedness
        d = torch.ones([self.B,3,3], device=xyz1.device)
        d[:,:,-1] = torch.sign(torch.det(V)*torch.det(W)).unsqueeze(1)

        # Rotation matrix U
        U = torch.matmul(d*V, W.permute(0,2,1)) # (B, 3, 3)
        
        if xyz1.dtype == torch.float16: #set ref to half
            U = U.to(torch.float16)

        return U
    @property
    def components(self):
        '''Componenets of the affine transform'''
        return self.centroid1, self.centroid2, self.U
    
    # Bona fide functions
    def apply(self, xyz):
        '''Apply the affine transform to xyz coordinates
        xyz (torch.tensor, (B, n, 3))
        '''
        return (xyz - self.centroid1) @ self.U + self.centroid2

def set_dist_to_target(net_xyz, sm, pdb_ref, pdb_rep):
    '''
    This repulsive loss works with sets of contigs.
    pdb_ref and pdb_rep should be set up so that each pair of contigs with its
    repulsive atoms are far apart (~200 A or more) from all other pairs. This ensures
    that when one set is aligned, the repulsive atoms from another set are
    not very likely to clash with parts of the hallucinated protein. (I find this
    less tedious than making a new ref/rep pdb pair for each contig.)
    
    Inputs
    ---------
    net_xyz (torch.tensor ((B, nres, 14, 3))): Predicted 3d coordinates of heavy atoms (only N, Ca and C are currently filled out)
    sm (SampledMask instance): Convenient way to translate between various indexes
    pdb_rep (dict): Parsed pdb. All protein heavy atoms are used for repulsion.
    '''
    
    # Housekeeping
    device = net_xyz.device
    B = 1
    
    # all hallucinated backbone atoms
    all_hal_xyz = net_xyz[:, :, :3, :]  # (B, res, NCaC, xyz)
    all_hal_xyz = all_hal_xyz.reshape(B, -1, 3)
    
    # get distances between each contig set bb atoms and pdb_rep fa
    dists = []
    for set_, cons in sm.set_to_con.items():
        # get idx0 for hal and ref
        set_ref_idx0 = []
        for con in cons:
            set_ref_idx0 += sm.map(con, 'ref', 'ref_idx0')
        set_ref_idx0 = torch.tensor(set_ref_idx0, dtype=torch.long)
        
        set_hal_idx0 = []
        for con in cons:
            set_hal_idx0 += sm.map(con, 'ref', 'hal_idx0')
        set_hal_idx0 = torch.tensor(set_hal_idx0, dtype=torch.long)
        
        # get N,Ca,C xyz positions
        set_ref_xyz = torch.tensor(pdb_ref['xyz'][set_ref_idx0, :3, :][None]).to(device)
        set_hal_xyz = net_xyz[:, set_hal_idx0, :3, :]  # (B, res, NCaC, xyz)
        set_ref_xyz = set_ref_xyz.view(B, -1, 3)  # (B, natom, xyz)
        set_hal_xyz = set_hal_xyz.view(B, -1, 3)  # (B, natom, xyz)
        
        m = pdb_rep['mask']
        pdb_rep_xyz = pdb_rep['xyz'][m]
        pdb_rep_xyz = torch.tensor(pdb_rep_xyz[None]).to(device)  # (B, natom, xyz)
        
        # get and apply affine transform
        affine_transform = AffineTransform(set_ref_xyz, set_hal_xyz)  # aligns anything in the ref frame to the hal frame
        pdb_rep_xyz_sup = affine_transform.apply(pdb_rep_xyz)  # pdb_rep now in the hal frame
        
        # pairwise distances between ALL_hal_xyz (bb) and pdb_rep (fa)
        d = (all_hal_xyz[:,:,None,:] - pdb_rep_xyz_sup[:,None,:,:]).pow(2).sum(-1).sqrt()  # (B, n_hal, n_rep)
        #d = d.view(B, -1)
        dists.append(d)
        
        # save for troubleshooting
        #torch.save(pdb_rep_xyz_sup.detach().cpu(), f'/home/dtischer/sandbox/211118/rep_xyz_sup_set_{set_}.pt')
        #ref_xyz_sup = affine_transform.apply(set_ref_xyz)
        #torch.save(ref_xyz_sup.detach().cpu(), f'/home/dtischer/sandbox/211118/ref_xyz_sup_set_{set_}.pt')
     
    dists = torch.cat(dists, dim=1)  # (B, n_hal*n_cons, n_rep)
    return dists
    
def calc_set_rep_loss(net_xyz, sm, pdb_ref, pdb_rep, sigma):
    d = set_dist_to_target(net_xyz, sm, pdb_ref, pdb_rep)
    return calc_lj_rep(d, sigma)

def calc_set_atr_loss(net_xyz, sm, pdb_ref, pdb_rep, sigma):
    d = set_dist_to_target(net_xyz, sm, pdb_ref, pdb_rep)
    return calc_lj_atr(d, sigma)
    
def rotation_matrix(xyz1, xyz2):
    '''
    Inputs
    -----------
    xyz1 (torch.tensor, (B, n_atom, 3))
    '''
    
    B = xyz1.shape[0]

    # center
    xyz1 -= xyz1.mean(1)
    xyz2 -= xyz2.mean(1)

    # Computation of the covariance matrix
    C = torch.matmul(xyz1.permute(0,2,1), xyz2)

    # Compute optimal rotation matrix using SVD
    V, S, W = torch.svd(C)

    # get sign to ensure right-handedness
    d = torch.ones([B,3,3], device=xyz1.device)
    d[:,:,-1] = torch.sign(torch.det(V)*torch.det(W)).unsqueeze(1)

    # Rotation matrix U
    U = torch.matmul(d*V, W.permute(0,2,1)) # (B, 3, 3)

    return U

def calc_rotation_loss(net_out, sm, ref_pdb):
    '''
    Currently only works if set 0 and set 1 should share some sort of symmetry.
    They must have the same number of residues. Be careful to list them
    in the corresponding order to --contigs.
    
    '''
    # Housekeeping
    device = net_out['c6d']['p_dist'].device
            
    # get idx0 of the two sets, indexed in the same order!
    set0_hal_idx0 = []
    for con in sm.set_to_con[0]:
        set0_hal_idx0 += sm.map(con, 'ref', 'hal_idx0')
    set0_hal_idx0 = torch.tensor(set0_hal_idx0, dtype=torch.long)
        
    set1_hal_idx0 = []
    for con in sm.set_to_con[1]:
        set1_hal_idx0 += sm.map(con, 'ref', 'hal_idx0')
    set1_hal_idx0 = torch.tensor(set1_hal_idx0, dtype=torch.long)
    
    set0_ref_idx0 = []
    for con in sm.set_to_con[0]:
        set0_ref_idx0 += sm.map(con, 'ref', 'ref_idx0')
    set0_ref_idx0 = torch.tensor(set0_ref_idx0, dtype=torch.long)
        
    # 1. SUPERIMPOSE HAL ONTO REF BY SET0 ONLY
    # (need to align to the ref first so that the z axis has meaning)
    # get N,Ca,C coordinates of set0 hal and ref
    set0_hal_xyz = net_out['xyz'][:, set0_hal_idx0, :3, :]
    set0_ref_xyz = torch.tensor(ref_pdb['xyz'][set0_ref_idx0, :3, :][None]).to(device)

    # get and apply transforms
    pred_centroid, ref_centroid, U = superimpose_pred_xyz(set0_hal_xyz, set0_ref_xyz)
    hal_xyz_aligned = (net_out['xyz'] - pred_centroid) @ U[:,None,:,:] + ref_centroid
    
    # spot check that the alignment is working
    #torch.save(net_out['xyz'].to('cpu'), 'hal_pre_alignment.pt')
    #torch.save(hal_xyz_aligned.to('cpu'), 'hal_post_alignment.pt')
    #x = torch.tensor(ref_pdb['xyz'][:, :3, :][None])
    #torch.save(x, 'ref_ptn.pt')
    
    # 2. **NOW** GET THE ROTATION MATRIX BETWEEN SET0 AND SET1
    # get N,Ca,C coordinates of set0 hal and set1 hal
    set0_hal_xyz = hal_xyz_aligned[:, set0_hal_idx0, :3, :]
    set1_hal_xyz = hal_xyz_aligned[:, set1_hal_idx0, :3, :]
    
    # reshape to (B=1,natom,3)
    set0_hal_xyz = set0_hal_xyz.view(1,-1,3)
    set1_hal_xyz = set1_hal_xyz.view(1,-1,3)
    
    # Apply L1 loss to "outside" of rotation matrix (rewards things that have an axis of rotation along the z axis)
    U = rotation_matrix(set0_hal_xyz, set1_hal_xyz)
    m = torch.tensor([[[0,0,1],
                       [0,0,1],
                       [1,1,1]]],
                     dtype=torch.bool
                    ).to(device)
    tgt_vals = torch.tensor([0, 0, 0, 0, 1]).to(device)
    loss_rot = torch.abs(U[m] - tgt_vals).sum()  # L1 loss to "snap" to desired values
    return loss_rot[None]  # add batch

def get_cce_loss(net_out, c6d_bins, mask=None, eps=1e-16):
    '''
    net_out - dict with keys: 'c6d' (all pairwise 6D transforms) and/or 'c3d' (3D cartesian coordinates).
                This function uses 'c6d' only. It is a dictionary with the following keys:
                keys: ['p_dist', 'p_omega', 'p_theta', 'p_phi']
                vals: (torch.float32) [batch_size, sequence_length, sequence_length, num_categories]
    c6d_bins - category label for each ij pair
                (torch.int32) [batch_size, sequence_length, sequence_length, 4]
    mask - 1.0 to include ij pair in cce calculation. The diagonal is automatically excluded.
            (torch.float32)[sequence_length, sequence_length]
    '''
    
    def cce(probs, labels, mask):
        """Calculates the categorical cross entropy and averages using a mask.
        Args:
          probs (torch.float32): [batch_size, sequence_length, sequence_length, num_categories]
          labels (torch.int32): [batch_size, sequence_length, sequence_length]
          mask (torch.float32): [batch_size, sequence_length, sequence_length]
        Returns:
          average_cce (torch.float32): [batch_size]
        """

        num_categories = probs.shape[-1]
        cce_ij = -torch.sum((F.one_hot(labels, num_categories)*torch.log(probs+eps))[:,:,:,:-1],axis=-1)
        cce_ave = torch.sum(mask*cce_ij, axis=(1,2)) / (torch.sum(mask, axis=(1,2)) + eps)
        return cce_ave
    
    # This loss function uses c6d
    dict_pred = net_out['c6d']
    
    # Housekeeping
    device = dict_pred['p_dist'].device
    B = dict_pred['p_dist'].shape[0]
    
    # Mask includes all ij pairs except the diagonal by default
    if mask is None:
        L_ptn = dict_pred['dist'].shape[1]
        mask = torch.ones(L_ptn)
    
    # Exclude diag
    L_mask = mask.shape[0]
    mask *= (1 - torch.eye(L_mask, dtype=torch.float32, device=mask.device))
    
    # Add batch
    mask = mask[None]
    mask = mask.to(device)
    
    # CCE loss
    cce_d = cce(dict_pred['p_dist'].to(device), c6d_bins[:,:,:,0].repeat(B,1,1).to(device), mask)
    cce_o = cce(dict_pred['p_omega'].to(device), c6d_bins[:,:,:,1].repeat(B,1,1).to(device), mask)
    cce_t = cce(dict_pred['p_theta'].to(device), c6d_bins[:,:,:,2].repeat(B,1,1).to(device), mask)
    cce_p = cce(dict_pred['p_phi'].to(device), c6d_bins[:,:,:,3].repeat(B,1,1).to(device), mask)
    loss = 0.25 * (cce_d + cce_o + cce_t + cce_p)
    
    return loss
   
def calc_dist_rmsd(pred_xyz, ref_xyz, mappings=None, log=False, eps=1e-6):
    '''
    Calculate distance RMSD
    Input:
        - pred_xyz: predicted coordinates (B, L, 3, 3)
        - ref_xyz: ref_xyz coordinates (B, L, 3, 3)
    Output: RMSD after superposition
    '''
    # extract constrained region
    if mappings is not None:
        pred_xyz = pred_xyz[:,mappings['con_hal_idx0'],:,:]
        ref_xyz = ref_xyz[mappings['con_ref_idx0'],:,:]

    ref_xyz = ref_xyz[None].to(pred_xyz.device)
    B, L = pred_xyz.shape[:2]

    pred_xyz = pred_xyz.reshape(B, L*3, 3)
    ref_xyz = ref_xyz.reshape(1, L*3, 3)

    D_pred_xyz = torch.triu(torch.cdist(pred_xyz, pred_xyz), diagonal=1) # (B, L*3, L*3)
    D_ref_xyz = torch.triu(torch.cdist(ref_xyz, ref_xyz), diagonal=1) # (1, L*3, L*3)
    n_pair = 0.5*(3*L*(3*L-1))

    diff_sq = torch.square(D_pred_xyz-D_ref_xyz).sum(dim=(-2, -1)) / n_pair
    rms_raw = torch.sqrt(diff_sq + eps)

    if log:
        rms = torch.log(rms_raw+1.0)
    else:
        rms = rms_raw
    
    return rms

def get_entropy_loss(net_out, mask=None, beta=1, dist_bins=37, eps=1e-16, type_='orig',k=5):
    '''
    net_out - dict with keys: 'c6d' (all pairwise 6D transforms) and/or 'c3d' (3D cartesian coordinates).
                This function uses 'c6d' only. It is a dictionary with the following keys:
                keys: ['p_dist', 'p_omega', 'p_theta', 'p_phi']
                vals: (torch.float32) [batch_size, sequence_length, sequence_length, num_categories]
    mask - 1.0 to include ij pair in cce calculation. The diagonal is automatically excluded.
            (torch.float32)[sequence_length, sequence_length]
    '''
    
    def entropy_orig(p, mask):
        S_ij = -(p * torch.log(p + eps)).sum(axis=-1)
        S_ave = torch.sum(mask * S_ij, axis=(1,2)) / (torch.sum(mask, axis=(1,2)) + eps)
        return S_ave
    
    def entropy_leaky(p,p_, mask):
        S_ij = -(p * torch.log(p_ + eps)[...,:-1]).sum(axis=-1)
        S_ave = torch.sum(mask * S_ij, axis=(1,2)) / (torch.sum(mask, axis=(1,2)) + eps)
        return S_ave
    
    def entropy_leaky_min(p,p_, mask):
        S_ij = -(p * torch.log(p_ + eps)[...,:-1]).sum(axis=-1)
        #pick top k
        S_ij = mask * S_ij
        S_ij[S_ij==0] = 100
        S_ij_min=torch.topk(S_ij, k,largest=False)[0]
        S_ij_min[S_ij_min==100] = 0 
        S_ave=torch.sum(S_ij_min,axis=(1,2))/ (torch.sum(mask, axis=(1,2)) + eps)*10 #*k
        return S_ave
    
    def entropy(p,p_, mask):
        if type_=='orig':
            return entropy_orig(p, mask)
        elif type_=='leaky':
            return entropy_leaky(p,p_, mask)
        elif type_=='leaky_min':
            return entropy_leaky_min(p,p_, mask)
    
    
    # This loss function uses c6d
    dict_pred = net_out['c6d']
    
    # Housekeeping
    device = dict_pred['p_dist'].device
    B = dict_pred['p_dist'].shape[0]

    # Mask includes all ij pairs except the diagonal by default
    if mask is None:
        L_ptn = dict_pred['p_dist'].shape[1]
        mask = torch.ones(L_ptn)
    
    # Exclude diag
    L_mask = mask.shape[0]
    mask *= (1 - torch.eye(L_mask, dtype=torch.float32, device=mask.device))
    
    # Add batch
    mask = mask[None]
    mask = mask.to(device)

    # Modulate sharpness of probability distribution
    # Also exclude >20A bin, which improves output quality
    pd = torch.softmax(torch.log(beta * dict_pred['p_dist'][...,:-1] + eps), axis = -1)
    pd_= torch.softmax(torch.log(beta * dict_pred['p_dist'] + eps), axis = -1)
    po = torch.softmax(torch.log(beta * dict_pred['p_omega'][...,:36] + eps), axis = -1)
    po_= torch.softmax(torch.log(beta * dict_pred['p_omega'] + eps), axis = -1)
    pt = torch.softmax(torch.log(beta * dict_pred['p_theta'][...,:36] + eps), axis = -1)
    pt_= torch.softmax(torch.log(beta * dict_pred['p_theta'] + eps), axis = -1)
    pp = torch.softmax(torch.log(beta * dict_pred['p_phi'][...,:18] + eps), axis = -1)
    pp_= torch.softmax(torch.log(beta * dict_pred['p_phi'] + eps), axis = -1)
    
    # Entropy loss  
    #print('distance')
    S_d = entropy(pd.to(device),pd_.to(device), mask)
    S_o = entropy(po.to(device),po_.to(device), mask)
    S_t = entropy(pt.to(device),pt_.to(device), mask)
    S_p = entropy(pp.to(device),pp_.to(device), mask)
    loss = 0.25 * (S_d + S_o + S_t + S_p)

    return loss

def get_kl_loss(net_out, bkg, mask=None, eps=1e-16):
    '''
    net_out - dict with keys: 'c6d' (all pairwise 6D transforms) and/or 'c3d' (3D cartesian coordinates).
                This function uses 'c6d' only. It is a dictionary with the following keys:
                keys: ['p_dist', 'p_omega', 'p_theta', 'p_phi']
                vals: (torch.float32) [batch_size, sequence_length, sequence_length, num_categories]
    mask - 1.0 to include ij pair in cce calculation. The diagonal is automatically excluded.
            (torch.float32)[sequence_length, sequence_length]
    bkg - background distributions for each c6d DOF
            dict with keys: ['p_dist', 'p_omega', 'p_theta', 'p_phi']
                      vals: (torch.float32) [batch_size, sequence_length, sequence_length, num_categories]
    '''

    def kl(p, q, mask, eps=1e-16):
        kl_ij = (p * torch.log(p/(q + eps) + eps)).sum(-1)
        kl_ij = torch.clamp(kl_ij, min=0, max=100)
        kl_ave = (mask * kl_ij).sum((1,2)) / (mask.sum((1,2)) + eps)
        return kl_ave
    
    # This loss function uses c6d
    dict_pred = net_out['c6d']
    
    # Housekeeping
    device = dict_pred['p_dist'].device
    B = dict_pred['p_dist'].shape[0]

    # Mask includes all ij pairs except the diagonal by default
    if mask is None:
        L_ptn = probs.shape[1]
        mask = torch.ones(L_ptn)
    
    # Exclude diag
    L_mask = mask.shape[0]
    mask *= (1 - torch.eye(L_mask, dtype=torch.float32, device=mask.device))
    
    # Add batch
    mask = mask[None]
    mask = mask.to(device) 
    
    # KL loss
    kl_d = kl(dict_pred['p_dist'].to(device), bkg['dist'].to(device), mask)
    kl_o = kl(dict_pred['p_omega'].to(device), bkg['omega'].to(device), mask)
    kl_t = kl(dict_pred['p_theta'].to(device), bkg['theta'].to(device), mask)
    kl_p = kl(dict_pred['p_phi'].to(device), bkg['phi'].to(device), mask)
    loss = -0.25 * (kl_d + kl_o + kl_t + kl_p)  # we are trying to MAXIMIZE the kl divergence
    
    return loss

def n_neighbors(xyz, n=1, m=9, a=0.5, b=2):
    """Gets the number of neighboring residues within a cone from each CA-CB vector.
    Inspired by formula from LayerSelector in RosettaScripts.
    
    Parameters
    ----------
        xyz : xyz coordinates of backbone atoms (batch, residues, atoms, xyz)
        n :   distance exponent
        m :   distance falloff midpoint
        a :   offset that controls the angular width of cone
        b :   angular sharpness exponent
    
    Returns
    -------
        n_nbr : number of neighbors (real-valued) for each position 
    """

    c6d, mask = geometry.xyz_to_c6d_smooth(xyz.permute(0,2,1,3),{'DMAX':20.0})

    dist = c6d[...,0]
    phi = c6d[...,3]

    f_dist = 1/(1+torch.exp(n*(dist*mask-m)))
    f_ang = ((torch.cos(np.pi-phi*mask)+a)/(1+a))**b
    n_nbr = torch.nansum(f_dist*f_ang*mask,axis=2)
    
    return n_nbr
 
def superimpose_pred_xyz(pred_xyz, ref_xyz, mappings=None):
    '''
    Returns the translation and rotation to superimpose pred_xyz on ref_xyz

    Input:
        - pred_xyz: predicted coordinates (B, L, 3, 3)
        - ref_xyz:  reference coordinates (B, L, 3, 3) I don't think this has batch dim
        - mappings: dictionary with keys 'con_ref_idx0' and 'con_hal_idx0' containing
                    the residue numbers of the motif to constrain

    Output:
        - pred_centroid: centroid of predicted motif (3)
        - ref_centroid:  centroid of reference motif (3)
        - U:             rotation matrix to align hallucinated motif with reference motif 
                         (after centering) (3, 3)
    '''
    
    pred_xyz=pred_xyz[:,:,:3,:]
    ref_xyz=ref_xyz[:,:3,:]
        
    def centroid(X):
        return X.mean(dim=-2, keepdim=True)

    # extract constrained region
    if mappings is not None:    
        pred_xyz = pred_xyz[:,mappings['con_hal_idx0'],:,:]
        ref_xyz = ref_xyz[mappings['con_ref_idx0'],:,:]
        

    ref_xyz = ref_xyz[None].to(pred_xyz.device)
    B, L = pred_xyz.shape[:2]

    # center to CA centroid
    pred_centroid = centroid(pred_xyz.view(B,3*L,3)).view(B,1,1,3)
    pred_xyz = pred_xyz - pred_centroid
    ref_centroid = centroid(ref_xyz.view(1,3*L,3)).view(1,1,1,3)
    ref_xyz = ref_xyz - ref_centroid

    # reshape ref_xyz crds to match the shape to pred_xyz crds
    pred_xyz = pred_xyz.view(B, L*3, 3)
    ref_xyz = ref_xyz.view(1, L*3, 3)
        
    if pred_xyz.dtype == torch.float16: #set ref to half
        ref_xyz=ref_xyz.to(torch.float16)
    
    # Computation of the covariance matrix
    C = torch.matmul(pred_xyz.permute(0,2,1), ref_xyz) 

    # Compute optimal rotation matrix using SVD
    V, S, W = torch.svd(C.to(torch.float32)) #SVD not implemented for half

    # get sign to ensure right-handedness
    d = torch.ones([B,3,3], device=pred_xyz.device)
    d[:,:,-1] = torch.sign(torch.det(V)*torch.det(W)).unsqueeze(1)

    # Rotation matrix U
    U = torch.matmul(d*V, W.permute(0,2,1)) # (B, 3, 3)  
    if pred_xyz.dtype == torch.float16: #set ref to half
        U = U.to(torch.float16)

    # Rotate pred_xyz
    rP = torch.matmul(pred_xyz, U) # (B, L*3, 3)
   
    return pred_centroid, ref_centroid, U 


def calc_crd_rmsd(pred_xyz, ref_xyz, mappings=None, log=False):
    '''
    Calculate coordinate RMSD

    Input:
        - pred_xyz: predicted coordinates of motif in reference frame (B, L*3, 3)
        - ref_xyz: reference coordinates of motif (B, L*3, 3)
    Output: RMSD
    '''
    pred_xyz=pred_xyz[:,:,:3,:]
    ref_xyz=ref_xyz[:,:3,:]
        
    def rmsd(V, W, eps=1e-6):
        L = V.shape[1]
        return torch.sqrt(torch.sum((V-W)*(V-W), dim=(1,2)) / L + eps)

    # extract constrained region
    if mappings is not None:    
        pred_xyz = pred_xyz[:,mappings['con_hal_idx0'],:,:]
        ref_xyz = ref_xyz[mappings['con_ref_idx0'],:,:]

    ref_xyz = ref_xyz[None].to(pred_xyz.device)
    B, L = pred_xyz.shape[:2]

    # reshape ref_xyz crds to match the shape to pred_xyz crds
    pred_xyz = pred_xyz.view(B, L*3, 3)
    ref_xyz = ref_xyz.view(1, L*3, 3)

    rms_raw = rmsd(pred_xyz, ref_xyz)

    if log:
        rms = torch.log(rms_raw+1.0)
    else:
        rms = rms_raw
        
    return rms

def get_dist_to_ligand(pred_xyz, lig_xyz):
    '''
    Calculate distances between predicted BB coordinates (superimposed to template frame) 
    to all heavy atoms in ligand.

    Input:
        - pred_xyz: predicted coordinates in reference frame (B, L, 3, 3)
        - lig_xyz:  ligand coordinates in reference frame (B, L_lig*3, 3)
    Output:
        - dist_lig: distances from predicted motif to ligand
    '''
    pred_xyz = pred_xyz[:,:,:3,:]
    B, L     = pred_xyz.shape[:2]
    pred_xyz = pred_xyz.reshape(B, L*3, 3)

    lig_xyz = lig_xyz[None].to(pred_xyz.device)
    #L_lig = lig_xyz.shape[1]
    #lig_xyz = lig_xyz.view(1, L_lig*3, 3)

    # calculate distances
    d = lig_xyz[:,:,None,:] - pred_xyz[:,None,:,:]
    dist = torch.linalg.norm(d,dim=3)
    return dist

def lj_rep(d, sigma, epsilon=1):
    '''
    Input:
        - d: distances from motif bb atoms to ligand atoms
        - sigma:    inter-atomic distance for repulsion
        - epsilon:  scale term for repulsion
    '''
    # Alford et al., JCTC 2017
    m = -12*epsilon/(0.6*sigma)*((1/0.6)**12 + (1/0.6)**6)
    E0 = epsilon*((1/0.6)**12 - 2*(1/0.6)**6 + 1)
    E_near = m*d + (E0-m*0.6*sigma)
    E_mid = epsilon*((sigma/d)**12 - 2*(sigma/d)**6 + 1)
    return (d<=0.6*sigma)*E_near + ((d>0.6*sigma) & (d<=sigma))*E_mid

def calc_lj_rep(lig_dist, sigma, epsilon=1):
    '''
    Calculate repulsion loss

    Input:
        - lig_dist: distances from motif bb atoms to ligand atoms
        - sigma:    inter-atomic distance for repulsion
        - epsilon:  scale term for repulsion
    Output:
        - loss:     Lennard-Jones-like repulsive loss
    '''
    loss = lj_rep(lig_dist, sigma=sigma, epsilon=epsilon)
    loss[torch.isnan(loss)] = 0

    #return torch.mean(loss, dim=[1,2])
    return torch.sum(loss, dim=[1,2]) / 1e5


def calc_lj_atr(lig_dist, sigma, epsilon=1):
    '''
    Calculate attraction loss

    Input:
        - lig_dist: distances from motif bb atoms to ligand atoms
        - sigma:    inter-atomic distance for repulsion
        - epsilon:  scale term for repulsion
    Output:
        - loss:     Lennard-Jones-like attractive loss
    '''
    def lj_atr(d, sigma, epsilon=1):
        # Alford et al., JCTC 2017
        E_mid = epsilon*((sigma/d)**12 - 2*(sigma/d)**6)
        return (d<=sigma)*(-epsilon) + (d>sigma)*E_mid

    loss = lj_atr(lig_dist, sigma=sigma, epsilon=epsilon)
    loss[torch.isnan(loss)] = 0

    #return torch.mean(loss, dim=[1,2])
    return torch.sum(loss, dim=[1,2]) / 1e5

def calc_rog(pred_xyz, thresh):
    ca_xyz = pred_xyz[:,:,1]
    sq_dist = torch.pow(ca_xyz - ca_xyz.mean(1),2).sum(-1).mean(-1)
    rog = sq_dist.sqrt()
    return F.elu(rog - thresh) + 1


class MultiLoss:
    def __init__(self):
        self.loss_funcs = {}
        self.weights = {}
        self.losses = {}
        self.precompute_funcs = []
        
    def __str__(self):
        # print all added loss functions and weights
        t = f'{"Loss term":<20}| {"Weight":<12}\n'
        for name, weight in self.weights.items():
            t += f'{name:<20}  {weight:.2f}\n'
            
        return t
        
        
    def add(self, name, loss_func, weight=1.):
        # loss_func must only be a function of the network prediction dictionary
        # Use a wrapper function (e.g. lambda) to pass all other args ahead of time
        if weight != -1:
            self.loss_funcs[name] = loss_func
            self.weights[name] = weight
            
    
    def remove(self, name):
        # Remove a loss function
        if (name in self.loss_funcs) and (name in self.weights):
            del self.loss_funcs[name]
            del self.weights[name]
        else:
            print(f'Loss function {name} has been removed from the loss.')
     
    def add_precompute(self, precompute_func):
        '''
        Add a function to compute quantities before computing losses.
        Results should be saved to net_out.
        '''
        self.precompute_funcs.append(precompute_func)

    
    def score(self, net_out):
        '''
        net_out: dictionary of outputs from the prediction network, plus any
        additional inputs your loss functions need. Add these additional inputs
        to net_out before passing it to this function, or add a function to
        compute them using `add_precompute`.

        '''
        # precompute any quantities needed by the loss terms
        for func in self.precompute_funcs:
            func(net_out)

        # compute loss terms and combine
        self.avg_loss = 0
        for name, func in self.loss_funcs.items():
            loss = func(net_out) # unweighted loss
            self.losses[name] = loss
            self.avg_loss += loss * self.weights[name]

        self.avg_loss /= sum(self.weights.values())

        return self.avg_loss
            
            
    def print_losses(self):
        # print pretty losses
        t = f'{"Loss Function":<20}| {"Unweighted":<12}| {"Weighted":<12}\n'

        for name, value in self.losses.items():
            loss_unweighted = value.detach().cpu().numpy()[0]
            loss_weighted = (self.weights[name] * value).detach().cpu().numpy()[0]          
            t += f'{name:<20}  {loss_unweighted: 6.3f}        {loss_weighted: 6.3f}\n'
            
        loss_avg = self.avg_loss.detach().cpu().numpy()[0]  
        t += f'Average loss: {loss_avg: 6.3f}\n'
        
        print(t)
 

    def print_header_line(self, column_width=12, **kwargs):
        '''Print loss names on a single line for tabular trajectory output'''

        line = f'{"avg loss":>{column_width}}'
        for name in self.weights.keys():
            line += f'{name:>{column_width}}'

        print(line, flush=True, **kwargs)


    def print_losses_line(self, column_width=12, mode='best', **kwargs):
        '''
        Print losses on a single line for tabular trajectory output
        
        Inputs
        ---------------
        mode: (<'best', 'all'>) If batch is >1, print the best result
                all results?
        
        '''
        if mode == 'best':
            batches = torch.tensor([self.avg_loss.argmin()])
        elif mode == 'all':
            batches = torch.arange(self.avg_loss.shape[0])
        # else just use the numbers in batches argument
            
        for i,b in enumerate(batches):
            if i != 0:
                line = column_width * ' '
            else:
                line = ''
            line += f'{float(self.avg_loss[b]):>{column_width}.4f}'
            for name, value in self.losses.items():
                line += f'{float(value[b]):>{column_width}.4f}'
            print(line, flush=True, **kwargs)
            
        if mode == 'all':
            n_cat = len(self.losses) + 1
            print(column_width * ' ' + n_cat * column_width * '-', flush=True)
            

def mk_bkg(Net, N, L, n_runs=100, net_kwargs={}):
    '''
    Inputs
    ------------
    
    
    Outputs
    ------------
    
    '''
    device = torch.device('cuda')
    B = 1
    logits = {k:[] for k in ['dist','omega','theta','phi']}
    
    for i in range(n_runs):
        inpt_cat = torch.randint(20, [B,N,L])  # don't include gaps in the fake sequences!! Hallucination never produces them!
        msa_one_hot = F.one_hot(inpt_cat, 21).type(torch.float32)
        out = Net(torch.argmax(msa_one_hot, axis=-1).to(device), 
                  msa_one_hot=msa_one_hot.to(device), 
                  **net_kwargs
                 )
        
        for k, v in logits.items():
            v.append(out[k].permute([0,2,3,1]))        
    
    # average logits
    logits = {k: torch.stack(v, axis=0).mean(0) for k, v in logits.items()}
    
    # probs
    probs = {k: F.softmax(v, dim=3) for k, v in logits.items()}
    
    return probs
