import numpy as np 
import util 
#from icecream import ic 


# extracts xyz and sequence, unlike RoseTTAFold.Perceiver_DMloss/parsers.py
def parse_pdb(filename):
    '''extract xyz coords for all heavy atoms'''
    lines = open(filename,'r').readlines()
    return parse_pdb_lines(lines)

def parse_pdb_lines(lines):
    # indices of residues observed in the structure
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]
    pdb_idx = [( l[21:22].strip(), int(l[22:26].strip()) ) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]  # chain letter, res num

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = l[21:22], int(l[22:26]), l[12:16], l[17:20]
        idx = pdb_idx.index((chain,resNo))
        for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
            if tgtatm == atom:
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    return {'xyz':xyz, # cartesian coordinates, [Lx14]
            'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
            'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
            'seq':np.array(seq), # amino acid sequence, [L]
            'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
           }

def parse_multichain_pdb(path):
    """
    Extract pdb info from file, along with chain information 
    """
    lines = open(path, 'r').readlines()
    return parse_multichain_lines(lines)

def parse_multichain_lines(lines):
    """
    Does the same thing as parse_pdb_lines but keeps track of where the chainbreaks are 

    Parameters:
        
        lines (list, required): Lines from pdb file 
    """

    # indices of residues observed in the structure
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]
    pdb_idx = [( l[21:22].strip(), int(l[22:26].strip()) ) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]  # chain letter, res num

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = l[21:22], int(l[22:26]), l[12:16], l[17:20]
        idx = pdb_idx.index((chain,resNo))
        for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
            if tgtatm == atom:
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    return {'xyz':xyz,              # cartesian coordinates, [Lx14]
            'mask':mask,            # mask showing which atoms are present in the PDB file, [Lx14]
            'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
            'chain':[i[0] for i in pdb_idx],             # chain identifiers for residues 
            'seq':np.array(seq),    # amino acid sequence, [L]
            'pdb_idx': pdb_idx,     # list of (chain letter, residue number) in the pdb file, [L]
            }

    
def getChainbreaks(pdict):
    """
    Get the locations of chain breaks in the PDB. Returns a list of indices where 
    a new chain begins. Note the list will always be at least length 1. 

    Parameters:
        
        pdict (dict, required): Dictionary containing the parsed pdb info 
    """
    breaks = []
    current_chain = None
    for i,chain in enumerate(pdict['chain']):
        if chain != current_chain:
            breaks.append(pdict['idx'][i])
            current_chain = chain

    return breaks

def get_idx(res_list): #idx in format 'A5,A6,A7,A8-10,A89' --> [('A', 5), ('A', 6), ('A', 7), ('A', 8), ('A', 9), ('A', 10), ('A', 89)]
    res_list=res_list.split(',')
    res_idx=[]
    for res in res_list:
        chain=res[0]
        res=res[1:].split('-')
        if len(res)==2:
            list_res=[*range(int(res[0]),int(res[1])+1)]
            for i in list_res:
                res_idx.append((chain,i))
        else:
            res_idx.append((chain,int(res[0])))
    return res_idx
