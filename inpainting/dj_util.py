import numpy as np 
import torch 
import pandas as pd
import random


def find_contigs(mask):
    """
    Find contiguous regions in a mask that are True with no False in between

    Parameters:
        mask (torch.tensor or np.array, required): 1D boolean array 

    Returns:
        contigs (list): List of tuples, each tuple containing the beginning and the  
    """
    assert len(mask.shape) == 1 # 1D tensor of bools 
    
    contigs = []
    found_contig = False 
    for i,b in enumerate(mask):
        
        
        if b and not found_contig:   # found the beginning of a contig
            contig = [i]
            found_contig = True 
        
        elif b and found_contig:     # currently have contig, continuing it 
            pass 
        
        elif not b and found_contig: # found the end, record previous index as end, reset indicator  
            contig.append(i)
            found_contig = False 
            contigs.append(tuple(contig))
        
        else:                        # currently don't have a contig, and didn't find one 
            pass 
    
    
    # fence post bug - check if the very last entry was True and we didn't get to finish 
    if b:
        contig.append(i+1)
        found_contig = False 
        contigs.append(tuple(contig))
        
    return contigs


def reindex_chains(pdb_idx):
    """
    Given a list of (chain, index) tuples, and the indices where chains break, create a reordered indexing 

    Parameters:
        
        pdb_idx (list, required): List of tuples (chainID, index) 

        breaks (list, required): List of indices where chains begin 
    """

    new_breaks, new_idx = [],[]
    current_chain = None

    chain_and_idx_to_torch = {}

    for i,T in enumerate(pdb_idx):

        chain, idx = T

        if chain != current_chain:
            new_breaks.append(i)
            current_chain = chain 
            
            # create new space for chain id listings 
            chain_and_idx_to_torch[chain] = {}
        
        # map original pdb (chain, idx) pair to index in tensor 
        chain_and_idx_to_torch[chain][idx] = i
        
        # append tensor index to list 
        new_idx.append(i)
    
    new_idx = np.array(new_idx)
    # now we have ordered list and know where the chainbreaks are in the new order 
    num_additions = 0
    for i in new_breaks[1:]: # skip the first trivial one
        new_idx[np.where(new_idx==(i+ num_additions*500))[0][0]:] += 500
        num_additions += 1
    
    return new_idx, chain_and_idx_to_torch,new_breaks[1:]

class ObjectView(object):
    '''
    Easy wrapper to access dictionary values with "dot" notiation instead
    '''
    def __init__(self, d):
        self.__dict__ = d

class SampledMask():
    def __init__(self, mask_str, ref_pdb_idxs, receptor_chain='?'):
        self.mask_str = mask_str
        self.ref_pdb_idxs = ref_pdb_idxs  # [(ch, res),...] of all reference residues
        self.inpaint_ranges = []
        self.receptor_chain = receptor_chain
        
    def __len__(self):
      return len(self.ref_pdb_idx)

    # declare dependent properties (can't directly set)
    @property
    def ref_pdb_idx(self):
        return self.expand(self.mask_str)

    @property
    def ref_pdb_ch(self):
        ch, res = zip(*self.ref_pdb_idx)
        return np.array(ch)

    @property
    def ref_pdb_res(self):
        ch, res = zip(*self.ref_pdb_idx)
        return np.array(res, dtype=float)
    
    @property
    def hal_idx0(self):
      return np.arange(self.__len__(), dtype=float)
    
    @property
    def hal_pdb_ch(self):
      return np.array(['B' if l==self.receptor_chain else 'A' for l in self.ref_pdb_ch])
    
    @property
    def hal_pdb_res(self):
      hal_pdb_ch = self.hal_pdb_ch.copy()  # avoiding multiple calls this way
      hal_pdb_res = np.full_like(hal_pdb_ch, 0., dtype=float)
      
      m_chA = hal_pdb_ch == 'A'
      L_chA = m_chA.sum()
      hal_pdb_res[m_chA] = np.arange(L_chA, dtype=float) + 1.
      
      m_chB = hal_pdb_ch == 'B'
      hal_pdb_res[m_chB] = self.ref_pdb_res[m_chB]
      return hal_pdb_res
    
    @property
    def ref_idx0(self):
      pdb_idx_to_idx0 = {pdb_idx: idx0 for idx0, pdb_idx in enumerate(self.ref_pdb_idxs)}
      return np.array([pdb_idx_to_idx0.get(pdb_idx) for pdb_idx in zip(self.ref_pdb_ch, self.ref_pdb_res.astype(int))], dtype=float)
    
    @property
    def idx_rf(self, gap=200):
      '''
      Residues indexes to be passed to RoseTTAFold. Any gaps in receptor
      numbering are preserved. A large number is added to the residue index
      upon each new chain.
      '''
      #idx_rf = np.arange(len(self.ref_pdb_ch))

      ## copy input receptor residue numbers (including any gaps)
      #mask = self.ref_pdb_ch==self.receptor_chain
      #idx_rf[mask] = self.ref_pdb_res[mask]

      ## add residue gaps between chains
      #chain_breaks = np.where(self.ref_pdb_ch[:-1] != self.ref_pdb_ch[1:])[0]+1 # chain starts
      #for i in chain_breaks:
      #    idx_rf[i:] += gap + idx_rf[i-1] - idx_rf[i] + 1

      #return idx_rf

      idx_rf = []
      idx = -1

      for i, (ch, res) in enumerate(zip(self.ref_pdb_ch, self.ref_pdb_res)):
        # note current ch (and "previous" ch if first row)
        if i == 0:
          if ch == self.receptor_chain:
            is_R, was_R = True, True
          else:
            is_R, was_R = False, False

          res_prev = res - 1
        else:
          is_R = ch == self.receptor_chain

        # increment idx
        if is_R != was_R:
          # chain jump
          idx += gap
        elif np.isnan(res_prev) or np.isnan(res):
          # increment by 1 in gap regions
          idx += 1
        else:
          # increment by jump in pdb numbering
          idx += res - res_prev

        # update was_R
        was_R = is_R
        res_prev = res
        idx_rf.append(idx)

      return np.array(idx_rf, dtype=float)
    
    @property
    def inpaint(self):
      '''
      Boolean array. True if sequence and structure should be rebuilt
      '''
      m_gap = np.isnan(self.ref_pdb_res)
      
      for src, sel in self.inpaint_ranges:
        m_range = self.mask_1d(src, sel)
        m_gap = m_gap | m_range
        
      return m_gap
    
    @property
    def df(self):
      '''
      Get a Pandas dataframe of all the properties
      '''
      data = {
        'hal_idx0': self.hal_idx0,
        'hal_pdb_ch': self.hal_pdb_ch,
        'hal_pdb_res': self.hal_pdb_res,
        'ref_idx0': self.ref_idx0,
        'ref_pdb_ch': self.ref_pdb_ch,
        'ref_pdb_res': self.ref_pdb_res,
        'idx_rf': self.idx_rf,
        'inpaint': self.inpaint,
      }
      
      return pd.DataFrame(data)
    
    @property
    def L_ref(self):
      return len(self.ref_pdb_idxs)
    
    @property
    def mask_contigs(self):
      '''
      True  if a residue is in a contig
      '''
      return ~np.isnan(self.ref_pdb_res)
    
    @property
    def hal_pdb_idx(self):
      return [(ch, res) for ch, res in zip(self.hal_pdb_ch, self.hal_pdb_res.astype(int))]
    
    @property
    def mappings(self):
      '''
      For backwards compatibiltiy.
      Here, 'contigs' are things NOT being inpainted
      '''
      m = ~self.inpaint.copy()  # don't recalculate when we call it multiple times
      
      con_hal_idx0 = self.hal_idx0[self.mask_contigs].astype(int)
      ref_hal_idx0 = self.ref_idx0[self.mask_contigs].astype(int)
      
      mappings = {
        'con_ref_pdb_idx': [(ch, res) for ch, res in zip(self.ref_pdb_ch[m], self.ref_pdb_res[m].astype(int))],
        'con_ref_idx0':    self.ref_idx0[m].astype(int),
        'con_hal_pdb_idx': [(ch, res) for ch, res in zip(self.hal_pdb_ch[m], self.hal_pdb_res[m].astype(int))],
        'con_hal_idx0':    self.hal_idx0[m].astype(int),
        'idx_rf':          self.idx_rf.astype(int),
        'hal_idx1':        self.hal_pdb_idx
      }
      return mappings
    
    #############################
    # methods
    #############################
    @staticmethod
    def parse_range(_range):
      if '-' in _range:
        s, e = _range.split('-')
      else:
        s, e = _range, _range

      return int(s), int(e)
 
    @staticmethod
    def parse_element(contig):
      '''
      Return the chain, start and end residue in a contig or gap str.

      Ex:
      'A4-8' --> 'A', 4, 8
      'A5'   --> 'A', 5, 5
      '4-8'  --> None, 4, 8
      'A'    --> 'A', None, None
      '''

      # is contig
      if contig[0].isalpha():
        ch = contig[0]
        if len(contig) > 1:
          s, e = SampledMask.parse_range(contig[1:])
        else:
          s, e = None, None
      # is gap
      else:
        ch = None
        s, e = SampledMask.parse_range(contig)

      return ch, s, e

    @staticmethod
    def contract(pdb_idx):
      # kind of like self, but defined locally for this function
      # (allows flush() to modify these variables, even though they're not passed)
      gvar = ObjectView(
        {'L_gap': 0,
         'con': [],
         'contracted': [],
        }
      )

      def flush():
        '''
        Appends a finished gap or con to `contracted` and then resets gaps and cons
        '''
        # write gap, then reset
        if gvar.L_gap > 0:
          gvar.contracted.append(str(gvar.L_gap))
          gvar.L_gap = 0

        # write contig, then reset
        if len(gvar.con) > 0:
          ch = gvar.con[0][0]
          s = gvar.con[0][1]
          e = gvar.con[-1][1]
          gvar.contracted.append(f'{ch}{s}-{e}')
          gvar.con = []

      for ch, res in pdb_idx:
        # new gap
        if (not ch) and (gvar.L_gap == 0) :
          flush()
          gvar.L_gap += 1

        # new con
        elif ch and (not gvar.con or res != res_prev + 1 or ch != ch_prev):  # (ch is alpha) (con == [])
          flush()
          gvar.con.append((ch, res))

        # extend gap
        elif (not ch) and (gvar.L_gap > 0):
          gvar.L_gap += 1

        # extend con
        elif (ch is not None) and (gvar.con):  # (ch is alpha) (con != [])
          gvar.con.append((ch, res))
          
        ch_prev = ch
        res_prev = res

      flush()
      return ','.join(gvar.contracted)
   
    @staticmethod
    def expand(mask_str):
        expanded = []

        for l in mask_str.split(','):
            ch, s, e = SampledMask.parse_element(l)

            # a contig
            if ch:  # if chain is something (not None)
                for res in range(s, e+1):
                    expanded.append((ch, res))

            # a gap
            else:
                for _ in range(s):
                    expanded.append((None, None))

        return expanded
      
    def mask_1d(self, src, sel):
      '''
      Convert selection to a boolean mask

      Input
      ----------
      sel (str): selection of a contig range or idx0 range. Can take multiple comma separated values of same type. Ex: A5-10,B2-8 or 3-8,14-21
      src (str): <'ref', 'hal'>
      '''
      # settle src ch, res and idx0 arrays
      if src == 'ref':
        a_ch = self.ref_pdb_ch
        a_res = self.ref_pdb_res
        a_idx0 = self.ref_idx0
      elif src == 'hal':
        a_ch = self.hal_pdb_ch
        a_res = self.hal_pdb_res
        a_idx0 = self.hal_idx0
      
      
      for l in sel.split(','):
        ch, s, e = SampledMask.parse_element(l)

        # selection type is pdb_idx
        if ch:
          m_ch = a_ch == ch
          m_res = (a_res >= s) & (a_res <= e)
          mask = m_ch & m_res
        # selection type is idx0
        else:
          mask = (a_idx0 >= s) & (a_idx0 <= e)
          
        return mask
      
    def add_inpaint_range(self, sel, src='ref'):
      self.inpaint_ranges.append((src, sel))
      
    def subsample(self, sel, src='ref'):
      '''
      Update mask_str that is a subset of the original mask_str
      Ex: self.mask_str = '2,A5-20,4', sel='A5-10' --> '2,A5-10,14'
      '''
      m = self.mask_1d(src, sel)
      
      expanded = []
      for b, ch, res in zip(m, self.ref_pdb_ch, self.ref_pdb_res.astype(int)):
        if b:
          expanded.append((ch, res))
        else:
          expanded.append((None, None))
          
      self.mask_str = self.contract(expanded)      
    
    def scatter_2d(self, ref_2d, fill_value=0.):
      '''
      Inputs
      ---------
      ref_2d (np.array; (L_ref, L_ref, ...)): Features to be scattered. The first two leading dimensions must be equal to L_ref.
      fill_value (float): Default value of array where ref_2d features are not scattered.
      '''
      assert ref_2d.shape[:2] == (self.L_ref, self.L_ref), f'ERROR: ref_2d must have leading dimensions of (L_ref, L_ref) ({self.L_ref, self.L_ref})'

      # make receiving array
      trailing_dims = ref_2d.shape[2:]
      dtype = ref_2d.dtype
      hal_2d = np.full((self.__len__(), self.__len__())+trailing_dims, fill_value=fill_value, dtype=dtype)

      # scatter
      con_hal_idx0 = self.hal_idx0[self.mask_contigs].astype(int)
      ref_hal_idx0 = self.ref_idx0[self.mask_contigs].astype(int)
      hal_2d[con_hal_idx0[:, None], con_hal_idx0[None, :]] = ref_2d[ref_hal_idx0[:, None], ref_hal_idx0[None, :]]

      return hal_2d
    
    def scatter_1d(self, ref_1d, fill_value=0.):
      '''
      Inputs
      ---------
      ref_1d (np.array; (L_ref, ...)): Features to be scattered. The first leading dimension must be equal to L_ref.
      fill_value (float): Default value of array where ref_1d features are not scattered.
      '''
      assert ref_1d.shape[0] == self.L_ref, f'ERROR: ref_1d must have leading dimension of L_ref ({self.L_ref})'
      
      # use numpy or torch?
      if type(ref_1d) is np.ndarray:
        obj = np
      elif type (ref_1d) is torch.Tensor:
        obj = torch

      # make receiving array
      trailing_dims = ref_1d.shape[1:]
      dtype = ref_1d.dtype
      hal_1d = obj.full((self.__len__(),)+trailing_dims, fill_value=fill_value, dtype=dtype)

      # scatter
      con_hal_idx0 = self.hal_idx0[self.mask_contigs].astype(int)
      ref_hal_idx0 = self.ref_idx0[self.mask_contigs].astype(int)
      hal_1d[con_hal_idx0] = ref_1d[ref_hal_idx0]

      # assign torch device
      if type (ref_1d) is torch.Tensor:
        hal_1d = hal_1d.to(ref_1d.device)

      return hal_1d

    def add_receptor(self, mask_str_rec, location='second'):
      '''
      Inputs
      ----------
      mask_str_rec (str):  ex: R12-100  (Receptor chain pdb_idxs should already be in self.ref_pdb_idxs)
      location <first, second>: Place receptor before or after hallucinated chain
      '''
      
      if location == 'first':
        self.mask_str = ','.join([mask_str_rec, self.mask_str])
      elif location == 'second':
        self.mask_str = ','.join([self.mask_str, mask_str_rec])
        
      self.receptor_chain = mask_str_rec[0]
      
    def set_receptor_chain(self, ch):
      self.receptor_chain = ch
      
    def change_ref(self, old_to_new, ref_pdb_idxs_new):
      '''
      Change the ref_pdb_idxs to something else. 
      
      Inputs
      -------------
      old_to_new (dict): Map (ch, res) in existing ref_pdb_idxs to new (ch, res) ref_pdb_idxs.
      ref_pdb_idxs_new ([(ch, res), ...]): New pdb_idxs to point to
      '''
      
      pdb_idx_new = [old_to_new.get(pdb_idx, (None,1)) for pdb_idx in self.ref_pdb_idx]
      self.mask_str = self.contract(pdb_idx_new)
      self.ref_pdb_idxs = ref_pdb_idxs_new      
      
    def copy(self):
      sm = SampledMask(self.mask_str, self.ref_pdb_idxs.copy(), self.receptor_chain)
      sm.inpaint_ranges = self.inpaint_ranges.copy()
      return sm

############################################################################################

# extra utils written by Joe Watchwell

############################################################################################

def translate_coords(parsed_pdb, res_translate):
    pdb_idx = parsed_pdb['pdb_idx']
    xyz = np.copy(parsed_pdb['xyz'])
    translated_coord_dict = {}
    for res in res_translate:
        res_idx = pdb_idx.index((res[0][0],int(res[0][1:]))) #get index in pdb of the chain and residue to be moved
        original_coords = np.copy(xyz[res_idx,:,:])
        init_dist = 1.01
        while init_dist > 1: #gives equal probability to any direction (as keeps going until init_dist is within unit circle)
            x = random.uniform(-1,1)
            y = random.uniform(-1,1)
            z = random.uniform(-1,1)
            init_dist = np.sqrt(x**2 + y**2 + z**2)
        x=x/init_dist
        y=y/init_dist
        z=z/init_dist
        x_trans = np.float32(x * random.uniform(0,float(res[1])))
        y_trans = np.float32(y * random.uniform(0,float(res[1])))
        z_trans = np.float32(z * random.uniform(0,float(res[1])))
        for i in range(14):
            if parsed_pdb['mask'][res_idx, i]:
                xyz[res_idx,i,0] += x_trans
                xyz[res_idx,i,1] += y_trans
                xyz[res_idx,i,2] += z_trans
        translated_coords = xyz[res_idx,:,:]
        translated_coord_dict[res[0]] = (original_coords, translated_coords)

    return xyz[:,:3,:], translated_coord_dict
