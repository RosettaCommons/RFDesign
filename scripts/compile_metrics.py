#!/usr/bin/env python
#
# Parses and compiles metrics previously computed by calc_metrics.sh.
#
# Usage: ./compile_metrics.py FOLDER
# 
# FOLDER should contain subfolders like lddt, trr_score, etc. This will output
# a file, combined_metrics.csv, in FOLDER.
#
import pandas as pd
import numpy as np
import os, glob, argparse, sys
from collections import OrderedDict

p = argparse.ArgumentParser()
p.add_argument('folder', help='Folder of outputs to process')
p.add_argument('--out',  help='Output file name.')
args = p.parse_args()

if args.out is None:
    args.out = os.path.join(args.folder,'combined_metrics.csv')

if not os.path.isdir(args.folder):
    sys.exit(f'ERROR: Input path {args.folder} not a folder.')

def parse_fastdesign_filters(folder):
    files = glob.glob(os.path.join(folder,'*.pdb'))
    records = []
    for f in files:
        row = OrderedDict()
        row['name'] = os.path.basename(f)[:-4]
        recording = False
        with open(f) as inf:
            for line in inf:
                if recording and len(line)>1:
                    tokens = line.split()
                    if len(tokens) == 2:
                        row[tokens[0]] = float(tokens[1])
                if '#END_POSE_ENERGIES_TABLE' in line: 
                    recording=True
                if line.startswith('pose'):
                    row['rosetta_energy'] = float(line.split()[-1])
        records.append(row)
    if len(records)>0: return pd.DataFrame.from_records(records)
    return pd.DataFrame({'name':[]})

def parse_lddt(folder):
    data = {'name':[], 'lddt':[]}
    files = glob.glob(os.path.join(folder,'*.npz'))
    if len(files)==0:
        return pd.DataFrame({'name':[]})
    for f in files:
        prefix = os.path.basename(f).replace('.npz','')
        lddt_data = np.load(f)
        data['lddt'].append(lddt_data['lddt'].mean())
        data['name'].append(prefix)
    return pd.DataFrame.from_dict(data)

def parse_rosetta_energy_from_pdb(folder):
    files = glob.glob(os.path.join(folder,'*.pdb'))
    records = []
    for pdbfile in files:
        with open(pdbfile) as inf:
            name = os.path.basename(pdbfile).replace('.pdb','')
            rosetta_energy = np.nan
            for line in inf.readlines():
                if line.startswith('pose'):
                    rosetta_energy = float(line.split()[-1])
            row = OrderedDict()
            row['name'] = name
            row['rosetta_energy'] = rosetta_energy
            records.append(row)
    if len(records)==0: return pd.DataFrame({'name':[]})
    return pd.DataFrame.from_records(records)

def parse_frag_qual(folder):
    records = []
    for frag_folder in glob.glob(os.path.join(folder,'*_fragments')):
        fn = os.path.join(frag_folder,'frag_qual.dat')
        if not os.path.exists(fn): continue
        with open(fn) as inf:
            lines = inf.readlines()
            index=1
            y_index=[]
            y_avg=[]
            y_bestmer=[]
            for line in lines:
                if int(line.split()[1]) == index:
                    y_index.append(float(line.split()[3]))
                else:
                    y_avg.append(np.average(np.array(y_index)))
                    y_bestmer.append(np.amin(np.array(y_index)))
                    y_index=[]
                index=int(line.split()[1])
            avg_all_frags=np.average(y_avg)
            avg_best_frags=np.average(y_bestmer)
        row = OrderedDict()
        row['name'] = os.path.basename(frag_folder).replace('_fragments','')
        row['avg_all_frags'] = avg_all_frags
        row['avg_best_frags'] = avg_best_frags
        records.append(row)
    if len(records)==0: return pd.DataFrame({'name':[]})
    return pd.DataFrame.from_records(records)

def parse_cce(folder):
    df = pd.DataFrame()
    for fn in glob.glob(os.path.join(folder,'*.trR_scored.txt')):
        row = pd.read_csv(fn)
        df = df.append(row)
    if df.shape[0]>0:
        df.columns = ['name','cce10','cce_1d','acc']
        return df[['name','cce10']]
    return pd.DataFrame({'name':[]})

def csv2df(fn,**kwargs):
    if os.path.exists(fn): return pd.read_csv(fn,**kwargs)
    return pd.DataFrame({'name':[]})

def parse_all_metrics(folder):
    df = pd.DataFrame({'name':[]})

    print(f'Parsing metrics in {folder}: ',end='')
    tmp = parse_lddt(os.path.join(folder,'lddt'))
    df = df.merge(tmp,on='name',how='outer')
    print(f'lddt ({tmp.shape[0]}), ',end='',flush=True)

    fn = os.path.join(folder,'pymol_metrics.csv')
    tmp = parse_fastdesign_filters(os.path.join(folder))
    df = df.merge(tmp,on='name',how='outer')
    print(f'rosetta metrics from PDB file ({tmp.shape[0]}), ',end='',flush=True)

    tmp = parse_cce(os.path.join(folder,'trr_score'))
    df = df.merge(tmp,on='name',how='outer')
    print(f'cce ({tmp.shape[0]}), ',end='',flush=True)

    tmp = parse_frag_qual(os.path.join(folder,'frags'))
    df = df.merge(tmp,on='name',how='outer')
    print(f'fragment quality ({tmp.shape[0]}), ',end='',flush=True)

    tmp = csv2df(os.path.join(folder,'tmscores.csv'),index_col=0)
    df = df.merge(tmp,on='name',how='outer')
    print(f'TM-scores ({tmp.shape[0]}), ',end='',flush=True)

    tmp = csv2df(os.path.join(folder,'pymol_metrics.csv'),index_col=0)
    df = df.merge(tmp,on='name',how='outer')
    print(f'Pymol metrics ({tmp.shape[0]}), ',end='',flush=True)

    tmp = csv2df(os.path.join(folder,'pyrosetta_metrics.csv'),index_col=0)
    df = df.merge(tmp,on='name',how='outer')
    print(f'Pyrosetta metrics ({tmp.shape[0]}), ',end='',flush=True)

    tmp = csv2df(os.path.join(folder,'af2_metrics.csv'),index_col=0)
    df = df.merge(tmp,on='name',how='outer')
    print(f'AlphaFold2 metrics ({tmp.shape[0]}), ',end='',flush=True)

    tmp = csv2df(os.path.join(folder,'../complex/interface_metrics.csv'),index_col=0)
    df = df.merge(tmp,on='name',how='outer')
    print(f'interface metrics ({tmp.shape[0]}), ',end='',flush=True)

    tmp = csv2df(os.path.join(folder,'rmsd_trr.csv'))
    if len(tmp)>0:
        df = df.merge(tmp[['name','rmsd']].rename(columns={'rmsd':'rmsd_trr'}),on='name',how='outer')
    print(f'TrR RMSD ({tmp.shape[0]}), ',end='',flush=True)

    tmp = csv2df(os.path.join(folder,'rmsd_trunk.csv'))
    if len(tmp)>0:
        df = df.merge(tmp[['name','rmsd']].rename(columns={'rmsd':'rmsd_trunk'}),on='name',how='outer')
    print(f'Trunk RMSD ({tmp.shape[0]})',flush=True)

    print(f'final dataframe shape: {df.shape}')
    print(f'final dataframe columns: {df.columns.values}')
    return df

df = parse_all_metrics(args.folder)
df.to_csv(args.out)
