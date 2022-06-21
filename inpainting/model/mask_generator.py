import torch
import sys
import random
from icecream import ic 
import numpy as np
from kinematics import xyz_to_t2d
from operator import itemgetter
#####################################
# Misc functions for mask generation
#####################################
def get_ss(ss_string):
    """
    Converts secondary structure string into a list of indices of where secondary structure changes (plus start and end coordinates of the sequence)
    """
    output = [0]
    for i in range(1,len(ss_string)):
        if ss_string[i] != ss_string[i-1]:
            output.append(i)
    if output[-1] != len(ss_string)-1:
        output.append(len(ss_string)-1)
    return output

def get_masks(ss, min_length, max_length, min_flank, max_flank):
    """
    Takes list of indices where secondary structure changes, and outputs coordinates for a masked region which masks out at least one whole secondary structure region, and allows flanking.
    """
    flank_width = random.randint(min_flank, max_flank)
    try:       
        options=[]
        for i in range(min_length, max_length+1):
            temp=[]
            for j in range(flank_width,ss[-1]-flank_width-i):
                if np.digitize([j,j+i], ss)[1] - np.digitize([j,j+i], ss)[0] > 1:
                    temp.append((j, j+i))
            options.append(temp)
        return random.choice(random.choice(options)), flank_width
        
    except:
        try:
            options=[]
            for i in range(min_length, max_length+1):
                temp=[]
                for j in range(flank_width,ss[-1]-flank_width-i):
                    if np.digitize([j,j+i], ss)[1] - np.digitize([j,j+i], ss)[0] > 0:
                        temp.append((j, j+i))
                options.append(temp)
            return random.choice(random.choice(options)), flank_width
        except:
            try:
                length = random.randint(min_length, max_length+1)
                start = random.randint(flank_width, ss[-1]-length-flank_width)
                return (start, start+length), flank_width
            except:
                return (4,7),3 #temporary fix to prevent crashing on small proteins

def get_double_masks(ss, min_length, max_length):
    '''
    Takes list of indices where secondary structure changes, and outputs coordinates for two masked regions which mask out one whole secondary structure region. No flanking.
    '''
    proportion = random.uniform(0.3,0.7)
    split_idx = int(np.round(ss[-1] * proportion,0))
    ss_1 = [i for i in ss if i < split_idx]
    ss_1.append(split_idx)
    ss_2 = [split_idx]
    ss_2.extend([i for i in ss if i > split_idx])
    split_1,_ = get_masks(ss_1, min_length, max_length,0,0)
    split_2,_ = get_masks(ss_2, min_length, max_length,0,0)
    ic(ss_1)
    ic(ss_2)
    return split_1, split_2

def get_double_masks_proximal(xyz_t,min_length,max_length,min_gap):
    temp_xyzt = torch.clone(xyz_t.unsqueeze(dim=0))
    t2d = xyz_to_t2d(temp_xyzt)[0,0,:,:,:37].numpy()
    t2d_flat = np.argmax(t2d, axis=-1)
    t2d_close = np.where(t2d_flat>11,0,1) #set anything less than 8A to 1, and more than to 0
    L = np.shape(t2d_close)[0]
    
    temp = []
    try:
        length1 = random.randint(np.minimum(min_length,L-min_gap),np.minimum(max_length,L-min_gap))
        length2 = random.randint(np.minimum(min_length,L-min_gap),np.minimum(max_length,L-min_gap))
        for j in range(0,L-length1-length2-min_gap):
            for k in range(min_gap+length2+j,L-length1):
                temp.append([j,k,np.mean(t2d_close[j:j+length1,k:k+length2])])
        sorted_temp = sorted(temp, key=itemgetter(2),reverse=True)
        choice = random.randint(0,int(0.05*len(sorted_temp)))
        output = sorted_temp[choice]
    except: #this is a temporary fix to handle small proteins
        length1 = 2
        length2 = 2
        output = [4,10,0.0] 
    return output, length1, length2

    

def get_rand_ss_idx(ss):
    """
    Takes list of indices where secondary structure changes, and outputs a single coordinate from within each ss element
    """
    output = []
    for i in range(len(ss)-1):
        if ss[i+1] - ss[i] > 4: #don't sample if ss region is smaller than 5 residues
            output.append(random.randint(ss[i],ss[i+1]-1))
    return output

def move_coords(xyz_t, translate_mask, dist_sd=1):
    """ Moves coordinates specified (False) in translate_mask.
        Moves by value picked from normal distribution (0 +/- sd = dist_sd)
    """
    B, N, L,res,coord = xyz_t.size()
    rand_mut = torch.stack((torch.randn(B,N,L,1).expand(B,N,L,3)*dist_sd,torch.randn(B,N,L,1).expand(B,N,L,3)*dist_sd,torch.randn(B,N,L,1).expand(B,N,L,3)*dist_sd),dim=-1)
    rand_mut[:,:,translate_mask.squeeze(),:,:] = 0
    xyz_t = torch.add(xyz_t, rand_mut)
    return xyz_t

#####################################
# Main mask generator function
#####################################

def generate_masks(msa, t1d, xyz_t, msa_mask_input, msa_full, task, ss, loader_params, mask_seq=None):
    '''
    Generates masks for different RF training tasks.
    separate into input masks and loss masks.
    all input masks are outputted as 1/True = unmasked, 0/False = masked.
    output masks are 1/True if scored, or 0/False if not.
    input masks are msa, t1d, str, t2d, t1d_coord_mask.
    loss masks are lddt, dih, bond, c6d, token.
    '''
    
    B,N,L = msa.size()
    t2d_b = xyz_t.shape[0]
    hal_mask_low = loader_params['HAL_MASK_LOW']
    hal_mask_high = loader_params['HAL_MASK_HIGH']
    flank_low = loader_params['FLANK_LOW']
    flank_high = loader_params['FLANK_HIGH']
    str2seqfull_low = loader_params['STR2SEQ_FULL_LOW']
    str2seqfull_high = loader_params['STR2SEQ_FULL_HIGH']
    constant_recycle = loader_params['CONSTANT_RECYCLE']

    #generate generic masks (blank masks)
    #input_masks
    t1d_mask = torch.full_like(t1d, True).bool()
    t1d_coord_mask = torch.full_like(t1d, False).bool()
    t1d_conf_mask = torch.full_like(t1d, 0.9) #scale confidences for non-fixedbbdes/hal tasks
    t2d_mask = torch.ones(t2d_b,L,L,44).bool() ### new t2ds are only 44 in last dimension (37+6+1)
    translate_mask = torch.ones(L).bool()
    msa_masktok_mask =  torch.ones((B,N,L), device=msa.device).bool() #mask for converting residues to mask token
    str_mask = torch.full_like(xyz_t, True).bool()
    
    #loss masks
    c6d_mask = torch.ones(B,37,L,L).bool() #[B,37,L,L]
    str_out_mask = torch.ones(L).bool()
    bond_mask = torch.ones(B,N,L).bool()
    dih_mask = lddt_mask = torch.ones(N,L).bool()
    
    
    
    if task == 'seq2str':
        '''
        Classic structure prediction task.
        '''
        #loss masks
        
        token_mask = msa_mask_input.clone()
        
    elif task == 'hal':
        '''
        This is Joe's hallucination task, where a contiguous region is masked, along with flanks, and two residues either end are given (but not their angles).
        The two floating points are randomly moved some distance.
        Scored on everything but the flank regions (may want to change this to only score central inpainted region
        '''
        ss_idx = get_ss(ss)
        splice, flank_width = get_masks(ss_idx, hal_mask_low, hal_mask_high, flank_low, flank_high)
        hal_mask_len = splice[1]-splice[0] 
        #input masks
        msa_masktok_mask[:,:,splice[0]-flank_width:splice[1]+flank_width] = False # flanks and central
        
        str_mask[:,splice[0]-flank_width:splice[1]+flank_width,:,:] = False # flanks and central
        str_mask[:,splice[0]-1,:,:] = True # immediate two flanking residues
        str_mask[:,splice[1],:,:] = True
        
        translate_mask[splice[0]-1] = False #imeediate two flanking residues
        translate_mask[splice[1]] = False
        
        t1d_mask[:,splice[0]-flank_width:splice[1]+flank_width,:] = False #flanks and central 
        # make mask to add to t1d to give 'fake' confidence in two floating coordinates
        t1d_coord_mask[:,splice[0]-1,:] = True # two immediate flanking residues
        t1d_coord_mask[:,splice[1],:] = True
        
        t1d_conf_mask = torch.full_like(t1d, 1) #set confidences to 1

        t2d_mask[:,:,splice[0]-1, 37:] = False #also masks last plane of last dimension
        t2d_mask[:,:,splice[1], 37:] = False
        t2d_mask[:,splice[0]-1, :, 37:] = False
        t2d_mask[:,splice[1], :, 37:] = False
        
        #loss masks
        randtemp = random.randint(0,10)
        
        token_mask = msa_mask_input.clone()
        token_mask[:,:,splice[0]:splice[1]] = True # central only
        bond_mask[:,:,splice[0]-flank_width:splice[0]] = False 
        bond_mask[:,:,splice[1]:splice[1]+flank_width] = False # flanks only
        dih_mask = lddt_mask = dih_mask * bond_mask[:,0,:] # flanks only
        c6d_mask[:,:,~dih_mask[0,:],:] = False
        c6d_mask[:,:,:,~dih_mask[0,:]] = False # flanks only
        
        str_out_mask[splice[0]-flank_width:splice[0]] = False
        str_out_mask[splice[1]:splice[1]+flank_width] = False #score on everything but flanks
    
    elif task == 'str2seq':
        '''
        This is Joe's str2seq task, where a contiguous region is masked, along with flanks.
        Everything, but the flanked regions, is scored
        ### This may need to change to upweight the central region, by only scoring on the central region.
        '''
        ss_idx = get_ss(ss)
        splice, flank_width = get_masks(ss_idx, hal_mask_low, hal_mask_high, flank_low, flank_high)
        hal_mask_len = splice[1]-splice[0] 
        
        #input masks
        msa_masktok_mask[:,:,splice[0]-flank_width:splice[1]+flank_width] = False # flanks and central
        
        str_mask[:,splice[0]-flank_width:splice[0],:,:] = False # flanks only
        str_mask[:,splice[1]:splice[1]+flank_width,:,:] = False
        
        t1d_mask[:,splice[0]-flank_width:splice[1]+flank_width,:] = False #flanks and central
        
        t1d_conf_mask = torch.full_like(t1d, 1) #set confidences to 1

        #loss masks
        token_mask = msa_mask_input.clone()
        token_mask[:,:,splice[0]:splice[1]] = True # central only
        
        bond_mask[:,:,splice[0]-flank_width:splice[0]] = False 
        bond_mask[:,:,splice[1]:splice[1]+flank_width] = False # flanks only

        dih_mask = lddt_mask = dih_mask * bond_mask[:,0,:] # flanks only
        c6d_mask[:,:,~dih_mask[0,:],:] = False
        c6d_mask[:,:,:,~dih_mask[0,:]] = False # flanks only
        
        str_out_mask[splice[0]-flank_width:splice[0]] = False
        str_out_mask[splice[1]:splice[1]+flank_width] = False #score on everything but flanks
    
    elif task == 'single_ss_hal':
        '''
        Here, a random amino acid is selected from each secondary structure element > 4 residues in length and unmasked.
        Angles of these residues ARE masked, so network just sees points in space.
        Scored on everything
        '''
        ss_idx = get_ss(ss)
        unmasked_coords = get_rand_ss_idx(ss_idx)
        #input masks
        msa_mask = torch.zeros((B,N,L), device=msa.device).bool() #mask out whole sequence
        msa_masktok_mask[:,:,:] = False      
        str_mask = ~str_mask
        str_mask[:,unmasked_coords,:,:] = True # give position of one amino acid in each ss element
        
        t1d_mask = ~t1d_mask
        t1d_mask[:,unmasked_coords,:] = True #### need to check what 1d features are
        
        t1d_conf_mask = torch.full_like(t1d, 1) #set confidences to 1
        
        
        t2d_mask[:,:,unmasked_coords, 37:] = False # mask out torsion angles
        t2d_mask[:,unmasked_coords, :, 37:] = False
        
        #loss masks
        token_mask = torch.full_like(msa_mask,   True).bool()
    
    elif task == 'str2seq_full':
        '''
        This is David's str2seq task, where most (default 90-100%) of sequence is masked.
        Score on everything.
        '''
        rand_prop = random.uniform(str2seqfull_low, str2seqfull_high) #random proportion masked, between two extremes  
        temp = torch.rand(msa_masktok_mask.shape, device=msa.device) < rand_prop #make mask
        if constant_recycle is True:
            temp = temp[:1,:,:].repeat(B,1,1)
        msa_masktok_mask[temp] = False
        temp2=temp.squeeze()
        t1d_mask[temp2] = False
        #loss masks
        token_mask = torch.full_like(msa_mask_input,   False).bool()
        token_mask[temp] = True

    elif task == 'packing_hal':
        '''
        This is a new task, where two regions are masked out so the network has to learn to do two-sided packing.
        '''
        #### v2, splices proximal parts of the protein
        
        splice, length1, length2 = get_double_masks_proximal(xyz_t,min_length=hal_mask_low,max_length=hal_mask_high,min_gap=20)
 
        #input masks
        msa_masktok_mask[:,:,splice[0]:splice[0]+length1] = False
        msa_masktok_mask[:,:,splice[1]:splice[1]+length2] = False

        str_mask[:,splice[0]:splice[0]+length1,:,:] = False
        str_mask[:,splice[1]:splice[1]+length2,:,:] = False

        t1d_mask[:,splice[0]:splice[0]+length1,:] = False
        t1d_mask[:,splice[1]:splice[1]+length2,:] = False
        
        t1d_conf_mask = torch.full_like(t1d, 1) #set confidences to 1
        #loss masks

        token_mask = msa_mask_input.clone()
        token_mask[:,:,splice[0]:splice[0]+length1] = True
        token_mask[:,:,splice[1]:splice[1]+length2] = True


    elif task == 'packing_termini':
        '''
        This is a new task, which masks the N and C termini, in the hope this will help the model pack a protein, without positional information about the termini.
        '''
        #choose length of N and C terminus mask
        split = int(np.round(L/2,0))
        if split < 10:
            term_N_len = 0
        elif L-split < 10:
            term_C_len = 0
        else:
            maxlen=min(40,split-10)
            term_N_len = random.randint(10,maxlen+1)
            maxlen=min(40,L-split-10)
            term_C_len = random.randint(10,maxlen+1)

        #input masks
        msa_masktok_mask[:,:,:term_N_len] = False
        msa_masktok_mask[:,:,L-term_C_len-1:] = False

        str_mask[:,:term_N_len,:,:] = False
        str_mask[:,L-term_C_len-1:,:,:] = False

        t1d_mask[:,:term_N_len,:] = False
        t1d_mask[:,L-term_C_len-1:,:] = False

        t1d_conf_mask = torch.full_like(t1d, 1) #set confidences to 1

        #loss masks

        token_mask = msa_mask_input.clone()
        token_mask[:,:,:term_N_len] = True
        token_mask[:,:,L-term_C_len-1:] = True

    elif task == 'msa_hal':
        '''
        This is Joe's hallucination task, where a contiguous region is masked, along with flanks, and two residues either end are given (but not their angles).
        This is adapted for msa inputs.
        The two floating points are randomly moved some distance.
        Scored on everything but the flank regions (may want to change this to only score central inpainted region
        '''
        ss_idx = get_ss(ss)
        splice, flank_width = get_masks(ss_idx, hal_mask_low, hal_mask_high, flank_low, flank_high)
        hal_mask_len = splice[1]-splice[0]
        #input masks
        msa_masktok_mask[:,:,splice[0]-flank_width:splice[1]+flank_width] = False # flanks and central

        str_mask[:,splice[0]-flank_width:splice[1]+flank_width,:,:] = False # flanks and central
        str_mask[:,splice[0]-1,:,:] = True # immediate two flanking residues
        str_mask[:,splice[1],:,:] = True

        translate_mask[splice[0]-1] = False #imeediate two flanking residues
        translate_mask[splice[1]] = False

        t1d_mask[:,splice[0]-flank_width:splice[1]+flank_width,:] = False #flanks and central
        # make mask to add to t1d to give 'fake' confidence in two floating coordinates
        t1d_coord_mask[:,splice[0]-1,:] = True # two immediate flanking residues
        t1d_coord_mask[:,splice[1],:] = True

        t1d_conf_mask = torch.full_like(t1d, 1) #set confidences to 1

        t2d_mask[:,:,splice[0]-1, 37:] = False #also masks last plane of last dimension
        t2d_mask[:,:,splice[1], 37:] = False
        t2d_mask[:,splice[0]-1, :, 37:] = False
        t2d_mask[:,splice[1], :, 37:] = False

        #loss masks
        randtemp = random.randint(0,10)

        token_mask = msa_mask_input.clone()
        token_mask[:,:,splice[0]:splice[1]] = True # central only
        bond_mask[:,:,splice[0]-flank_width:splice[0]] = False
        bond_mask[:,:,splice[1]:splice[1]+flank_width] = False # flanks only
        dih_mask = lddt_mask = dih_mask * bond_mask[0,:,:] # flanks only. This line differs from 'hal' task masks
        c6d_mask[:,:,~dih_mask[0,:],:] = False
        c6d_mask[:,:,:,~dih_mask[0,:]] = False # flanks only

        str_out_mask[splice[0]-flank_width:splice[0]] = False
        str_out_mask[splice[1]:splice[1]+flank_width] = False #score on everything but flanks

    elif task in ['stub_fixedbb', 'stub_centroid', 'stub_fixedseq']:
        '''
        This is the stub design task for a fixed backbone.
        '''
        ic(mask_seq)
        ic(mask_seq.dtype)
        token_mask = msa_mask_input.clone()
        if task in ['stub_fixedbb', 'stub_centroid']:
            msa_masktok_mask[:,:,mask_seq] = False
            t1d_mask[:,mask_seq,:] = False
            token_mask[:,:,mask_seq] = True
    
        if task in ['stub_fixedseq', 'stub_centroid']:
            '''
            This is the stub design task where only the centroid of the stub is given.
            '''
            str_mask[:,mask_seq,:,:] = False
            # N, L,res,coord = xyz_t.size()
            ic(xyz_t.shape)
            stub_start = torch.min(mask_seq)
            stub_len = len(mask_seq)
            mask_seq_continguous = torch.arange(stub_start, stub_start+stub_len, dtype=mask_seq.dtype)
            torch.testing.assert_close(mask_seq, mask_seq_continguous)
            stub_center_idx = mask_seq_continguous[stub_len // 2]
            # C_BETA = 2
            # stub_centroid = xyz_t[:,C_BETA,:]
            # stub_centroid = stub_crop.c_alpha_centroid(xyz_t[0, mask_seq])
            # CENTROID_NOISE = 0
            # stub_centroid += stub_crop.spherical_sample(CENTROID_NOISE)
            # xyz_t[0, stub_center_idx, stub_crop.C_ALPHA_INDEX] = stub_centroid
            str_mask[:, stub_center_idx, :] = True
            # ic('str_mask idxs:', (~str_mask).nonzero())

    else:
        sys.exit(f'Masks cannot be generated for the {task} task!')
    
    #horrible logic, but mask2s mask out the whole column (one hot encodings and indel rows in msa or msa_full), and then masks bring back the mask token in planes 21 and 42.

    msa_masked_mask = torch.ones(B,N,L,46).bool()
    msa_masked_mask2 = msa_masktok_mask.unsqueeze(-1)
    msa_masked_mask2 = msa_masked_mask2.repeat(1,1,1,46)
    msa_masked_mask[:,:,:,20] = msa_masktok_mask
    msa_masked_mask[:,:,:,42] = msa_masktok_mask
    
    #make msa_full_mask compatible with all tasks
    full_N = msa_full.size()[1]
    msa_full_mask2 = msa_masked_mask2[:,:1,:,:23].repeat(1,full_N,1,1)
    msa_full_mask = msa_masked_mask[:,:1,:,:23].repeat(1,full_N,1,1)
    
    
    mask_dict = {'t1d_mask':t1d_mask,
                't1d_coord_mask':t1d_coord_mask,
                't1d_conf_mask':t1d_conf_mask,
                't2d_mask':t2d_mask,
                'str_mask':str_mask,
                'msa_masktok_mask':msa_masktok_mask,
                'msa_masked_mask':msa_masked_mask,
                'msa_masked_mask2':msa_masked_mask2,
                'msa_full_mask':msa_full_mask,
                'msa_full_mask2':msa_full_mask2,
                'c6d_mask':c6d_mask,
                'lddt_mask':lddt_mask,
                'dih_mask':dih_mask,
                'bond_mask':bond_mask,
                'token_mask':token_mask,
                'translate_mask':translate_mask,
                'str_out_mask':str_out_mask
                }
    
    return mask_dict
