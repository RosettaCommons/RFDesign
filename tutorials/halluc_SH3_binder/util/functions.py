import os,sys,glob,shutil
import numpy as np
import pandas as pd

def split_folder(folder, batch_size, trb_dir=None, start_batch=0, ndigits=None):
    '''Splits files in a folder into subdirectories with <batch_size> files each.'''

    N = batch_size
    i = 0
    b = start_batch
    if trb_dir is None:
        trb_dir = folder+'/../'
        
    filenames = sorted(glob.glob(folder+'*pdb'))
    if ndigits is None:
        ndigits = int(np.ceil(np.log10(len(filenames)/N)))

    for fn in filenames:
        subfolder = f'{b:0{ndigits}}/'
        os.makedirs(folder+'/'+subfolder, exist_ok=True)
        shutil.move(fn, folder+'/'+subfolder+os.path.basename(fn))
        trbfn = os.path.basename(fn).replace('.pdb','.trb')
        if os.path.islink(folder+'/'+trbfn):
            os.symlink(trb_dir+'/'+trbfn, folder+'/'+subfolder+trbfn)
            os.remove(folder+'/'+trbfn)

        i += 1
        if i>=N:
            b += 1
            i = 0

def get_trb_info(trb):
    idx = trb['con_hal_pdb_idx']
    templ_con = f'A{idx[0][1]}-{idx[-1][1]}'

    idxmap = dict(zip([x[1] for x in trb['con_ref_pdb_idx']], [x[1] for x in trb['con_hal_pdb_idx']]))
    force_aa = []
    for x in trb['settings']['force_aa'].split(','):
        aa = x[-1]
        i = int(x[1:-1])
        force_aa.append(f'A{idxmap[i]}'+aa)
    force_aa = ','.join(force_aa)
    return templ_con, force_aa

def load_af2_int_metrics(folder):
    tmp = pd.read_csv(folder+'out.sc',delim_whitespace=True)
    tmp['folder'] = folder
    tmp = tmp.rename(columns={'description':'name'})
    tmp['name'] = tmp['name'].str.replace('_af2pred','')
    if os.path.exists(folder+'/relax/interface_metrics.csv'):
        tmp2 = pd.read_csv(folder+'/relax/interface_metrics.csv')
        tmp2['name'] = tmp2['name'].apply(lambda x: x.replace('_af2pred_0001',''))
        tmp = tmp.merge(tmp2,on='name')
    if os.path.exists(folder+'/binder_rmsd.csv'):
        tmp2 = pd.read_csv(folder+'/binder_rmsd.csv')
        tmp = tmp.merge(tmp2,on='name')
    return tmp

def load_seqs(folder):
    records = []
    for fn in glob.glob(folder+'/*fas'):
        with open(fn) as f:
            lines = f.readlines()
            seq = lines[1].strip()
            if '/' in seq:
                seq = seq[:seq.index('/')]
        records.append(dict(
            name=os.path.basename(fn).replace('.fas',''),
            seq=seq
        ))
    return pd.DataFrame.from_records(records)

def load_mpnn_seqs(folder, run=None, offset=0):
    records = []
    for fn in glob.glob(folder+'*.fa'):
        name = os.path.basename(fn).replace('.fa','')
        with open(fn) as f:
            lines = f.readlines()
        for i,x in enumerate(lines[3::2]):
            if run is not None:
                name_ = name+f'_{run}'
            records.append(dict(
                name=name_+f'_{i+offset}',
                seq=x.strip()))
    return pd.DataFrame.from_records(records)

def get_dist_matrix(hits):
    dist = np.full((len(hits),len(hits)), np.nan)
    for i in range(len(hits)):
        for j in range(i):
            a = np.array(list(hits.iloc[i]['seq']))
            b = np.array(list(hits.iloc[j]['seq']))
            if len(a) != len(b):
                dist[i,j] = 0
            else:
                dist[i,j] = sum(a==b)/len(a)
            dist[j,i] = dist[i,j]
    return dist

def filter_on_seq_id(df, thresh = 0.8, verbose=False):
    names = list(df['name'])
    i = 0
    while i < len(names)-1:
        seq_i = df[df['name']==names[i]]['seq'].values[0]
        seq_list_i = np.array(list(seq_i))
        L = len(seq_i)

        j = i+1
        while j < len(names):
            seq_j = df[df['name']==names[j]]['seq'].values[0]
            if len(seq_list_i)==len(seq_j):
                fid = sum(seq_list_i==np.array(list(seq_j)))/L
            else:
                fid = 0
            if fid > thresh:
                if verbose: print(names[i], names[j], fid)
                names = names[:j]+names[j+1:]
            else:
                j += 1
        i += 1  
    return names
