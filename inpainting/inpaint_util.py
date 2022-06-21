import numpy as np 
import torch 
import pandas as pd
import random
import math
import sys
from icecream import ic 

num2aa=[
    'ALA','ARG','ASN','ASP','CYS',
    'GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO',
    'SER','THR','TRP','TYR','VAL',
    'UNK']

aa2num= {x:i for i,x in enumerate(num2aa)}

aa1to3 =  \
    {'C': 'CYS', 'D': 'ASP', 'S': 'SER', 'Q': 'GLN', 'K': 'LYS',
     'I': 'ILE', 'P': 'PRO', 'T': 'THR', 'F': 'PHE', 'N': 'ASN',
     'G': 'GLY', 'H': 'HIS', 'L': 'LEU', 'R': 'ARG', 'W': 'TRP',
     'A': 'ALA', 'V':'VAL', 'E': 'GLU', 'Y': 'TYR', 'M': 'MET',
     '?': 'UNK'}

aa3to1 = {val:key for key,val in aa1to3.items()}

# full sc atom representation (Nx14)
aa2long=[
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # ala
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None), # arg
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None), # asn
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None), # asp
    (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None), # gln
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None), # glu
    (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
    (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None), # his
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None), # ile
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None), # leu
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None), # lys
    (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None), # met
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None), # phe
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None), # pro
    (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
    (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # thr
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE2"," CE3"," NE1"," CZ2"," CZ3"," CH2"), # trp
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None), # tyr
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # val
]


def seq2chars(seq):
    
    out = []
    for char in seq:
        three = num2aa[char]
        one   = aa3to1[three]

        out.append(one)

    return ''.join(out)

        
class ObjectView(object):
    '''
    Easy wrapper to access dictionary values with "dot" notiation instead
    '''
    def __init__(self, d):
        self.__dict__ = d

class ResidueMap():
    def __init__(self, contig_list, ref_pdb_idxs,
                 inpaint_seq_ranges=[], inpaint_str_ranges=[]
                ):


        self.contig_list    = contig_list  # MUST PASS A LIST!
        self.contig_string  = ','.join(contig_list)
        self.ref_pdb_idxs   = ref_pdb_idxs  # [(ch, res),...] of all reference residues
        self.inpaint_seq_ranges = inpaint_seq_ranges
        self.inpaint_str_ranges = inpaint_str_ranges
        self.hal_chain_order = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
    def __len__(self):
      return len(self.ref_pdb_idx)

    # declare dependent properties (can't directly set)
    @property
    def ref_pdb_idx(self):
      return self.expand(self.contig_string)
    
    @property
    def ref_pdb_ch(self):
      ch, res = zip(*self.ref_pdb_idx)
      return np.array(ch)
    
    @property
    def ref_pdb_res(self):
      ch, res = zip(*self.ref_pdb_idx)
      return np.array(res, dtype=float)
    
    @property
    def ref_idx0(self):
      pdb_idx_to_idx0 = {pdb_idx: idx0 for idx0, pdb_idx in enumerate(self.ref_pdb_idxs)}
      return np.array([pdb_idx_to_idx0.get(pdb_idx) for pdb_idx in zip(self.ref_pdb_ch, self.ref_pdb_res.astype(int))], dtype=float)
    
    def use_original_pdb_idx(self, con_str):
      return (None, None) not in self.expand(con_str)
    
    @property
    def hal_pdb_ch(self):
      # Ensure new chain letters won't conflict with chains for which we wish to keep the pdb numbering
      ch_letters_used = [con_str[0] for con_str in self.contig_list if self.use_original_pdb_idx(con_str)]
      ch_letters_free = [ch for ch in self.hal_chain_order if ch not in ch_letters_used]
      
      chains = []
      for con_str in self.contig_list:
        if self.use_original_pdb_idx(con_str):
          ch = con_str[0]
        else:
          ch = ch_letters_free[0]
          ch_letters_free.pop(0)
        
        L = len(self.expand(con_str))
        chains.append(np.array(L * [ch]))
        
      return np.concatenate(chains)
    
    @property
    def hal_pdb_res(self):
      residues = []
      for con_str in self.contig_list:
        if self.use_original_pdb_idx(con_str):
          ch = con_str[0]
          res = self.ref_pdb_res[self.hal_pdb_ch == ch]
        else:
          L = len(self.expand(con_str))
          res = np.arange(L) + 1.
        residues.append(res)
        
      return np.concatenate(residues)
          
    @property
    def hal_pdb_idx(self):
      return [(ch, res) for ch, res in zip(self.hal_pdb_ch, self.hal_pdb_res.astype(int))]
    
    @property
    def hal_idx0(self):
      return np.arange(self.__len__(), dtype=float)
    
    @property
    def idx_rf(self, gap=200):
      '''
      Residues indexes to be passes to RoseTTAFold. Index jumps by `gap` when their is a new chain.
      '''
      idx_rf = []
      idx = -1

      for i, (ch, res) in enumerate(zip(self.hal_pdb_ch, self.ref_pdb_res)):
        # note current ch (and "previous" ch if first row)
        if i == 0:
          ch_prev = ch
          res_prev = res - 1

        # chain jump
        if ch != ch_prev:
          idx += gap
        # increment by 1 in gap regions
        elif np.isnan(res_prev) or np.isnan(res):
          idx += 1
        # increment by jump in pdb numbering
        else:
          idx += res - res_prev

        # update was_R
        ch_prev = ch
        res_prev = res
        idx_rf.append(idx)

      return np.array(idx_rf, dtype=float)
    
    @property
    def inpaint_seq(self):
      '''
      Boolean array. True if sequence should be rebuilt
      '''
      m_gap = np.isnan(self.ref_pdb_res)
      for src, sel in self.inpaint_seq_ranges:
        m_range = self.mask_1d(src, sel)
        m_gap = m_gap | m_range
      return m_gap
    
    @property
    def inpaint_str(self):
      '''
      Boolean array. True if structure should be rebuilt
      '''
      m_gap = np.isnan(self.ref_pdb_res)
      for src, sel in self.inpaint_str_ranges:
        m_range = self.mask_1d(src, sel)
        m_gap = m_gap | m_range
      return m_gap
    
    @property
    def inpaint_both(self):
      '''
      Boolean array. True if sequence and structure should be rebuilt
      '''
      return self.inpaint_str & self.inpaint_seq
    
    @property
    def df(self):
      '''
      Get a Pandas dataframe of all the properties
      '''
      data = {
        'hal_idx0':     self.hal_idx0,
        'hal_pdb_ch':   self.hal_pdb_ch,
        'hal_pdb_res':  self.hal_pdb_res,
        'ref_idx0':     self.ref_idx0,
        'ref_pdb_ch':   self.ref_pdb_ch,
        'ref_pdb_res':  self.ref_pdb_res,
        'idx_rf':       self.idx_rf,
        'inpaint_seq':  self.inpaint_seq,
        'inpaint_str':  self.inpaint_str,
        'inpaint_both': self.inpaint_both,
        'inpaint_k':    self.inpaint_k,
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
    def mappings(self):
      '''
      For backwards compatibiltiy.
      Here, 'contigs' are things NOT being inpainted
      '''
      m = ~self.inpaint_str.copy()  # don't recalculate when we call it multiple times
      
      con_hal_idx0 = self.hal_idx0[self.mask_contigs].astype(int)
      ref_hal_idx0 = self.ref_idx0[self.mask_contigs].astype(int)

      mappings = {
        'con_ref_pdb_idx': [(ch, res) for ch, res in zip(self.ref_pdb_ch[m], self.ref_pdb_res[m].astype(int))],
        'con_ref_idx0':    self.ref_idx0[m].astype(int),

        'con_hal_pdb_idx': [(ch, res) for ch, res in zip(self.hal_pdb_ch[m], self.hal_pdb_res[m].astype(int))],
        'con_hal_idx0':    self.hal_idx0[m].astype(int),

        'idx_rf':          self.idx_rf.astype(int),
        'hal_idx1':        self.hal_pdb_idx,
        'mask':            m,
        'inpaint_str':      self.inpaint_str
      }
      return mappings
    
    @property
    def inpaint_k(self):
      '''
      Get the current inpaint_k mask
      '''
      if hasattr(self, '_inpaint_k'):
        return self._inpaint_k
      else:
        return np.zeros(len(self), dtype=bool)
    
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
          s, e = ResidueMap.parse_range(contig[1:])
        else:
          s, e = None, None
      # is gap
      else:
        ch = None
        s, e = ResidueMap.parse_range(contig)

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
    def expand(contig_string):
        expanded = []

        for l in contig_string.split(','):
            ch, s, e = ResidueMap.parse_element(l)

            # a contig
            if ch:  # if chain is something (not None)
                for res in range(s, e+1):
                    expanded.append((ch, res))

            # a gap
            else:
                for _ in range(s):
                    expanded.append((None, None))
        return expanded
      
    @staticmethod
    def con_len(contig_string):
      return len(self.expand(contig_string))
      
    ###########################
    # Instance methods
    ###########################
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
      
      mask = np.full_like(a_ch, False, dtype=bool)
      for l in sel.split(','):
        ch, s, e = ResidueMap.parse_element(l)
        # selection type is pdb_idx
        if ch:
          m_ch = a_ch == ch
          m_res = (a_res >= s) & (a_res <= e)
          mask_update = m_ch & m_res
        # selection type is idx0
        else:
          mask_update = (a_idx0 >= s) & (a_idx0 <= e)
          
        mask = mask | mask_update
          
      return mask
      
    def subsample(self, sel, src='ref'):
      '''
      Update contig_string that is a subset of the original contig_string
      Ex: self.contig_string = '2,A5-20,4', sel='A5-10' --> '2,A5-10,14'
      '''
      m = self.mask_1d(src, sel)
      
      expanded = []
      for b, ch, res in zip(m, self.ref_pdb_ch, self.ref_pdb_res.astype(int)):
        if b:
          expanded.append((ch, res))
        else:
          expanded.append((None, None))
          
      self.contig_string = self.contract(expanded)      
    
    def scatter_1d(self, ref_1d, fill_value=0., feature='str'):
      '''
      Inputs
      ---------
      ref_1d (np.array; (L_ref, ...)): Features to be scattered. The first leading dimension must be equal to L_ref.
      fill_value (float): Default value of array where ref_1d features are not scattered.
      feature (str): <str, seq> Should the inpaint range mask for STRucture or SEQuence be applied?
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
      m_inpaint = self.__getattribute__(f'inpaint_{feature}')
      m = self.mask_contigs & ~m_inpaint
      con_hal_idx0 = self.hal_idx0[m].astype(int)
      ref_hal_idx0 = self.ref_idx0[m].astype(int)
      hal_1d[con_hal_idx0] = ref_1d[ref_hal_idx0]

      # assign torch device
      if type (ref_1d) is torch.Tensor:
        hal_1d = hal_1d.to(ref_1d.device)

      return hal_1d
    
    def scatter_2d(self, ref_2d, fill_value=0., feature='str'):
      '''
      Inputs
      ---------
      ref_2d (np.array; (L_ref, L_ref, ...)): Features to be scattered. The first two leading dimensions must be equal to L_ref.
      fill_value (float): Default value of array where ref_2d features are not scattered.
      feature (str): <str, seq> Should the inpaint range mask for STRucture or SEQuence be applied?
      '''
      assert ref_2d.shape[:2] == (self.L_ref, self.L_ref), f'ERROR: ref_2d must have leading dimensions of (L_ref, L_ref) ({self.L_ref, self.L_ref})'

      # make receiving array
      trailing_dims = ref_2d.shape[2:]
      dtype = ref_2d.dtype
      hal_2d = np.full((self.__len__(), self.__len__())+trailing_dims, fill_value=fill_value, dtype=dtype)

      # scatter
      m_inpaint = self.__getattribute__(f'inpaint_{feature}')
      m = self.mask_contigs & ~m_inpaint
      con_hal_idx0 = self.hal_idx0[m].astype(int)
      ref_hal_idx0 = self.ref_idx0[m].astype(int)
      
      print(con_hal_idx0, ref_hal_idx0)
      hal_2d[con_hal_idx0[:, None], con_hal_idx0[None, :]] = ref_2d[ref_hal_idx0[:, None], ref_hal_idx0[None, :]]

      return hal_2d
    
    def change_ref(self, old_to_new, ref_pdb_idxs_new):
      '''
      Change the ref_pdb_idxs to something else. 
      
      Inputs
      -------------
      old_to_new (dict): Map (ch, res) in existing ref_pdb_idxs to new (ch, res) ref_pdb_idxs.
      ref_pdb_idxs_new ([(ch, res), ...]): New pdb_idxs to point to
      '''
      
      pdb_idx_new = [old_to_new.get(pdb_idx, (None,1)) for pdb_idx in self.ref_pdb_idx]
      self.contig_string = self.contract(pdb_idx_new)
      self.ref_pdb_idxs = ref_pdb_idxs_new      
      
    def copy(self):
      rm = ResidueMap(self.contig_string, self.ref_pdb_idxs.copy(), self.receptor_chain)
      rm.inpaint_ranges = self.inpaint_ranges.copy()
      return sm

    def sample_inpaint_k(self, k, src='both'):
      '''
      Set a boolean mask with k entries that are True in the given inpaint range
      k (int): Number of positions to inpaint
      src (str, <seq, str, both>): Parent inpaint range to subsample from
      '''
      mask_parent = self.__getattribute__(f'inpaint_{src}')
      idxs = np.where(mask_parent)[0]
      idxs_k = np.random.choice(idxs, k, replace=False)
      mask_k = np.full_like(mask_parent, False, dtype=bool)
      mask_k[idxs_k] = True
      self._inpaint_k = mask_k
      
############################################################################################

# extra utils written by Joe Watchwell

############################################################################################

def get_translated_coords(args):
    '''
    Parses args.res_translate
    '''
    #get positions to translate
    res_translate = []
    for res in args.res_translate.split(":"):
        temp_str = []
        for i in res.split(','):
            temp_str.append(i)
        if temp_str[-1][0].isalpha() is True:
            temp_str.append(2.0) #set default distance
        for i in temp_str[:-1]:
            if '-' in i:
                start = int(i.split('-')[0][1:])
                while start <= int(i.split('-')[1]):
                    res_translate.append((i.split('-')[0][0] + str(start),float(temp_str[-1])))
                    start += 1
            else:
                res_translate.append((i, float(temp_str[-1])))
        start = 0
    
    output = []
    for i in res_translate:
        temp = (i[0], i[1], start)
        output.append(temp)
        start += 1

    return output

def get_tied_translated_coords(args, untied_translate=None):
    '''
    Parses args.tie_translate
    '''
    #pdb_idx = list(parsed_pdb['idx'])
    #xyz = parsed_pdb['xyz']
    #get positions to translate
    res_translate = []
    block = 0
    for res in args.tie_translate.split(":"):
        temp_str = []
        for i in res.split(','):
            temp_str.append(i)
        if temp_str[-1][0].isalpha() is True:
            temp_str.append(2.0) #set default distance
        for i in temp_str[:-1]:
            if '-' in i:
                start = int(i.split('-')[0][1:])
                while start <= int(i.split('-')[1]):
                    res_translate.append((i.split('-')[0][0] + str(start),float(temp_str[-1]), block))
                    start += 1
            else:
                res_translate.append((i, float(temp_str[-1]), block))
        block += 1
    
    #sanity check
    if untied_translate != None:
        checker = [i[0] for i in res_translate]
        untied_check = [i[0] for i in untied_translate]
        for i in checker:
            if i in untied_check:
                print(f'WARNING: residue {i} is specified both in --res_translate and --tie_translate. Residue {i} will be ignored in --res_translate, and instead only moved in a tied block (--tie_translate)')
        
        final_output = res_translate
        for i in untied_translate:
            if i[0] not in checker:
                final_output.append((i[0],i[1],i[2] + block + 1))
    else:
        final_output = res_translate
    
    return final_output

 

def translate_coords(parsed_pdb, res_translate):
    '''
    Takes parsed list in format [(chain_residue,distance,tieing_block)] and randomly translates residues accordingly.
    '''

    pdb_idx = parsed_pdb['pdb_idx']
    xyz = np.copy(parsed_pdb['xyz'])
    translated_coord_dict = {}
    #get number of blocks
    temp = [int(i[2]) for i in res_translate]
    blocks = np.max(temp)

    for block in range(blocks + 1):
        init_dist = 1.01
        while init_dist > 1: #gives equal probability to any direction (as keeps going until init_dist is within unit circle)
            x = random.uniform(-1,1)
            y = random.uniform(-1,1)
            z = random.uniform(-1,1)
            init_dist = np.sqrt(x**2 + y**2 + z**2)
        x=x/init_dist
        y=y/init_dist
        z=z/init_dist
        translate_dist = random.uniform(0,1) #now choose distance (as proportion of maximum) that coordinates will be translated
        for res in res_translate:
            if res[2] == block:
                res_idx = pdb_idx.index((res[0][0],int(res[0][1:])))
                original_coords = np.copy(xyz[res_idx,:,:])
                for i in range(14):
                    if parsed_pdb['mask'][res_idx, i]:
                        xyz[res_idx,i,0] += np.float32(x * translate_dist * float(res[1]))
                        xyz[res_idx,i,1] += np.float32(y * translate_dist * float(res[1]))
                        xyz[res_idx,i,2] += np.float32(z * translate_dist * float(res[1]))
                translated_coords = xyz[res_idx,:,:]
                translated_coord_dict[res[0]] = (original_coords.tolist(), translated_coords.tolist())
         
    return xyz[:,:3,:], translated_coord_dict

def parse_block_rotate(args):
    block_translate = []
    block = 0
    for res in args.block_rotate.split(":"):
        temp_str = []
        for i in res.split(','):
            temp_str.append(i)
        if temp_str[-1][0].isalpha() is True:
            temp_str.append(math.pi/18) #set default angle to 10 degrees
        for i in temp_str[:-1]:
            if '-' in i:
                start = int(i.split('-')[0][1:])
                while start <= int(i.split('-')[1]):
                    block_translate.append((i.split('-')[0][0] + str(start),float(temp_str[-1]), block))
                    start += 1
            else:
                block_translate.append((i, float(temp_str[-1]), block))
        block += 1
    return block_translate

def rotate_block(xyz, block_rotate,pdb_index):
    rotated_coord_dict = {}
    #get number of blocks
    temp = [int(i[2]) for i in block_rotate]
    blocks = np.max(temp)
    for block in range(blocks + 1):
        idxs = [pdb_index.index((i[0][0],int(i[0][1:]))) for i in block_rotate if i[2] == block]
        angle = [i[1] for i in block_rotate if i[2] == block][0]
        block_xyz = xyz[idxs,:,:]
        com = [float(torch.mean(block_xyz[:,:,i])) for i in range(3)]
        print('com', com)
        origin_xyz = np.copy(block_xyz)
        for i in range(np.shape(origin_xyz)[0]):
            for j in range(3):
                origin_xyz[i,j] = origin_xyz[i,j] - com
        rotated_xyz = rigid_rotate(origin_xyz,angle,angle,angle)
        recovered_xyz = np.copy(rotated_xyz)
        for i in range(np.shape(origin_xyz)[0]):
            for j in range(3):
                recovered_xyz[i,j] = rotated_xyz[i,j] + com
        recovered_xyz=torch.tensor(recovered_xyz)
        rotated_coord_dict[f'rotated_block_{block}_original'] = block_xyz
        rotated_coord_dict[f'rotated_block_{block}_rotated'] = recovered_xyz
        xyz_out = torch.clone(xyz)
        for i in range(len(idxs)):
            xyz_out[idxs[i]] = recovered_xyz[i]
    return xyz_out,rotated_coord_dict
        

def rigid_rotate(xyz,a=math.pi,b=math.pi,c=math.pi):
    alpha = random.uniform(-a, a)
    beta = random.uniform(-b, b)
    gamma = random.uniform(-c, c)
    rotated = []
    for i in range(np.shape(xyz)[0]):
        for j in range(3):
            x = xyz[i,j,0]
            y = xyz[i,j,1]
            z = xyz[i,j,2]
            x2 = x*math.cos(alpha) - y*math.sin(alpha)
            y2 = x*math.sin(alpha) + y*math.cos(alpha)
            x3 = x2*math.cos(beta) - z*math.sin(beta)
            z2 = x2*math.sin(beta) + z*math.cos(beta)
            y3 = y2*math.cos(gamma) - z2*math.sin(gamma)
            z3 = y2*math.sin(gamma) + z2*math.cos(gamma)
            rotated.append([x3,y3,z3])

    rotated=np.array(rotated)
    rotated=np.reshape(rotated, [np.shape(xyz)[0],3,3])
    
    return rotated

