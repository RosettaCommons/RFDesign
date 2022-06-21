import sys, os
import time
import numpy as np
import torch
import torch.nn as nn

# distributed data parallel
from icecream import ic
import util


sys.path.append('./model')
import RoseTTAFoldModel

SS2IDX = {'L':0, # change SS id to integer 
          'E':1,
          'H':2,
          'U':3}

IDX2SS = {val:key for key,val in SS2IDX.items()}

def ss_to_tensor(ss):
    """
    Turn secondary structure annotations into integers 

    Parameters:
        ss (numpy.ndarray): Array with dtype'<U1' (strings?)

    Returns:
        ss_out: torch.tensor 
    """
    ss_out = []
    for i_ss, assignment in enumerate(ss):

        ss_out.append(SS2IDX[assignment])   # convert ss assignment to integer 
    
    return nn.functional.one_hot( torch.tensor(ss_out), num_classes=4)


def zero_ss_information(ss_feats, zero_inf_mask, zero_inf_state):
    """
    Apply the zero information state to any SS tokens with True in zero_inf_mask

    """
    #NOTE: This function broken if num ss dimensions is 3 instead of 4 

    N = zero_inf_mask.sum() # total number of positions that need zero information 
    
    if zero_inf_state == 'unknown':
        # apply the unknown token (4) to all positions which are True 
        replacement = nn.functional.one_hot( torch.full((N,), 3), num_classes=4).float()

    elif zero_inf_state == 'uniform':
        # apply uniform distribution over SS where True 
        a = 1/3
        replacement = torch.tensor([a,a,a,0.])[None].repeat(N,1)

    ss_feats[zero_inf_mask] = replacement 

    return ss_feats 


def get_chunk_boundaries(L, max_chunks, min_chunks=1):
    """
    Get indices of the boundaries that would divide an array into chunks of random size 
    
    Parameters:
        L (int, required): Length of protein (or crop)
        
        max_chunks (int, required): Maximum number of chunks to make 
        
        min_chunks (int, optional): Min number of chunks 
    """
    assert (max_chunks > 0) and (min_chunks > 0)
    
    N = np.random.randint(min_chunks, max_chunks) # random number of chunks 
    
    # number of boundaries is 1 less than number of chunks
    # L-1 so that when we add 1 later we don't have boundaries at non-existent indices
    boundaries = torch.randperm(L-1)[:N] 
    sorted_, indices = torch.sort(boundaries)
    
    # add 1 to ensure we don't have a boundary at 0
    return sorted_ + 1


def smear_ss_chunks(ss_hot, boundaries):
    """
    Smear secondary structure content according to boundaries
    """
    assert len(ss_hot.shape) == 2 # (L,?)
    ss_smeared = ss_hot.clone()
    N = len(boundaries)

    # fence post bug 1: end can never be 0, else error 
    start=0
    for i,end in enumerate(boundaries):
        end = int(end)
        
        chunk = ss_hot[start:end]
        avg_chunk = torch.mean(chunk, dim=0) # mean ss features along the chunk

        
        ss_smeared[start:end] = avg_chunk 
        
        start = end 
    
    # fence post bug 2: do the last remaining chunk 
    chunk = ss_hot[end:]
    avg_chunk = torch.mean(chunk, dim=0)
    ss_smeared[end:] = avg_chunk
    
    return ss_smeared


def sample_swap_ss(orig_ss_hot, use_unknown=False, frac_mut=0.25):
    """
    Given the original one-hot encoding of the secondary structure, "mutate" certain positions
    to have RANDOM secondary structure

    Parameters:
        orig_ss_hot (torch.tensor, required): Original one hot encoding of secondary sructure

        use_unkown (bool, optional): If true, allowed to sample "uknown" ss tokens, else just helix/sheet/loop

        frac_mut (float, optional): Fraction of residues to randomly sample SS for
    """
    new_ss_hot = torch.clone(orig_ss_hot)

    classes = 4 if use_unknown else 3
    L = new_ss_hot.shape[0]

    # create the uniform distribution over L residues
    dist = torch.ones(L,classes)
    div  = dist.sum(dim=-1)[:,None]
    dist /= div

    # take a single sample at each position and one hot encode
    sampled_ss_hot = nn.functional.one_hot( torch.multinomial(dist, 1).squeeze(), num_classes=4).float()

    mask = torch.rand(L) < frac_mut # bool mask

    # swap the sampled ss at positions where mask is true
    new_ss_hot[mask] = sampled_ss_hot[mask]

    return new_ss_hot


def maskAndCatExtra1D(t1d, extra_t1d, mask_dict, chosen_task):
    """
    Takes extra 1D features, masks them, and concatenates them to existing 1D features 

    Parameters:
        t1d (torch.tensor, required): Traditional 22 dim 1D template features 

        extra_t1d (dict, required): Dict of extra 1d feature tensors

        mask_dict (dict, required): Dictionary of masks 

        chosen_task (str, required): current task, in case masking should be conditioned on it 
    """
    t1d_out = [t1d] # list of tensors to append to and then concat 
    
    extra_keys = list(extra_t1d.keys())
    for key in extra_keys:


        # (1) mask the features 
        feats = extra_t1d[key]
        mask  = mask_dict[key]
        
        # Mask must come in a form which can be added 
        # to perform the desired operation 
        feats = feats + mask  

        # a bit hacky: check if there is a mismatch in the second dimension
        #              This occurs in seq2str mode, 
        if (t1d.shape[1] != feats[None,...].shape[1]):
            repeats = t1d.shape[1]
            feats = feats.repeat(repeats,1,1)


        t1d_out.append(feats[None,...]) # expand first dim to be able to concat 

    
    t1d_out = torch.cat(t1d_out, dim=-1)

    return t1d_out 


def init_lecun_normal(shape, scale=1.0):
    def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
        normal = torch.distributions.normal.Normal(0, 1)

        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma

        alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
        p = alpha_normal_cdf + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform

        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
        x = torch.clamp(x, a, b)

        return x

    def sample_truncated_normal(shape, scale=1.0):
        stddev = np.sqrt(scale/shape[-1])/.87962566103423978  # shape[-1] = fan_in
        return stddev * truncated_normal(torch.rand(shape))

    out_param = torch.nn.Parameter( (sample_truncated_normal(shape)) )
    return out_param

def init_zeros(shape):

    out_param = torch.nn.Parameter( torch.zeros(shape)  )
    return out_param


def custom_ckpt_load(model, ckpt, special_keys=[], init_method=init_lecun_normal, init_kwargs={}):

    """
    Manually loads parameters into model with support for mismatch in tensor sizes 
    for specified modules within modules in module_keys

    Parameters:
        model (torch.nn.module, required): RoseTTAFold model

        ckpt (dict, required): Loaded torch checkpoint dict 


    """
    print('From inside custom ckpt load, special keys: ',special_keys)
    pretrained_state = ckpt['model_state_dict'] # state from pretrained model 
    model_state = model.state_dict()            # state from initialized model with different architecture 

    param_keys = list(model_state.keys())       # iter through all keys in the model which is being initialized 
                                                # to ensure we don't miss any keys 

    for i_key,key in enumerate(param_keys):

        if 'module.' in key:
            key_safe = key.replace('module.','')
        else:
            key_safe = key 
        
        # if the key isn't one of the special ones, load like normal 
        if key_safe not in special_keys:

            model_state[key] = pretrained_state[key_safe]
        
        # key is special, support the different shapes of the params 
        else:
            print(f'Found special parameter {key}, accomodating shape mismatch')

            shape_pretrained = pretrained_state[key_safe].shape
            shape_model      = model_state[key].shape

            # for now, only allow model to be larger than pretrained 
            for i_dim,_ in enumerate(shape_pretrained):
                assert shape_pretrained[i_dim] <= shape_model[i_dim]
            
            # create a new tensor whose first entries in 
            # every dimension are the pretrained ones
            pretrained_param = pretrained_state[key_safe]
            new_param        = init_method(shape_model,**init_kwargs) 

            
            # replace params in new tensor with pretrained ones 
            if len(shape_model) == 2: # rank 2 (matrix)
                a,b = shape_pretrained
                new_param.data[:a,:b] = pretrained_param


            elif len(shape_model) == 1: # rank 1 (vector)
                a = shape_pretrained[0]
                new_param.data[:a] = pretrained_param


            else:
                raise RuntimeError('Cannot currently custom load parameters with number of dims = {len(shape_model)}')

            # put new tensor into pretrained model 
            model_state[key] = new_param

    model.load_state_dict(model_state)
    print('Successful custom checkpoint load')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            

MODEL_PARAM = {'SE3_param': {'div': 4,
                                'l0_in_features': 32,
                                'l0_out_features': 32,
                                'l1_in_features': 3,
                                'l1_out_features': 2,
                                'n_heads': 4,
                                'num_channels': 32,
                                'num_degrees': 2,
                                'num_edge_features': 32,
                                'num_layers': 3},
                  'd_hidden': 32,
                  'd_hidden_templ': 64,
                  'd_msa': 256,
                  'd_msa_full': 64,
                  'd_pair': 128,
                  'd_templ': 64,
                  'n_head_msa': 8,
                  'n_head_pair': 4,
                  'n_head_templ': 4,
                  'n_module_2track': 24,
                  'n_module_3track': 8,
                  'p_drop': 0.15}

if __name__ == '__main__':
    # ADD T1D features here:
    standard_1d = 21+1
    ss_1d    = 3
    bonus_1d = 1
    dim_t1d = standard_1d + ss_1d + bonus_1d

    MODEL_PARAM['d_t1d'] = dim_t1d

    model_extra = RoseTTAFoldModel.RoseTTAFoldModule(**MODEL_PARAM)
    print('Number of model parameters is ',count_parameters(model_extra))

    #ckpt_path = '/home/minkbaek/for/jue/RoseTTAFold.Nov05/models/BFF_last.pt'
    ckpt_path = 'models/BFF_last.pt'
    has_gpu = torch.cuda.is_available()
    if not has_gpu:
        kwargs={'map_location':'cpu'}
    else:
        kwargs={}
    ckpt = torch.load(ckpt_path,**kwargs)


    custom_ckpt_load(model_extra, ckpt, special_keys=['templ_emb.emb.weight', 'templ_emb.emb.bias'], init_method=init_zeros)
