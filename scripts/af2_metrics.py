#!/software/conda/envs/SE3/bin/python -u
# 
# Makes AlphaFold2 predictions and computes various RMSDs between AF2 models,
# hallucination design models, and reference structures.
#
# Usage (on a GPU node):
#
#     ./af2_metrics.py FOLDER
#
# This outputs AF2 models to FOLDER/af2/ and metrics to FOLDER/af2_metrics.csv.
# The script automatically uses a template PDB found in the .trb file
# corresponding to each design. If you would like to specify the template, you
# can do:
#
#   ./af2_metrics.py --template TEMPLATE_PDB FOLDER
#
# Passing a file with space-delimited residue numbers (as numbered in the
# reference structure) to --interface_res will also output an RMSD only on
# these positions.
#
#     ./pyrosetta_metrics.py --template TEMPLATE_PDB --interface_res
#     RESNUM_FILE FOLDER
#
# Updated 2021-8-13

import os, sys, argparse, glob, time

import numpy as np
import pandas as pd
from collections import OrderedDict

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir+'/../hallucination/models/alphafold/')
from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.relax import relax

from jax.lib import xla_bridge
from alphafold.model.tf import shape_placeholders                                                        
import tensorflow.compat.v1 as tf

os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0'

sys.path.append(script_dir+'/../hallucination/util/')
from parsers import parse_pdb
from contigs import parse_contigs
import util


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('input_data', help='Folder of TrDesign outputs to process, or a single pdb file')
    p.add_argument('-t','--template', help='Template (natural binder) structure (.pdb)')
    p.add_argument('--template_dir', help='Template (natural binder) directory')
    # p.add_argument('--sc_rmsd', action='store_true', default=True,
    #     help='Also calculate side-chain RMSD, returning NaN if residues aren\'t matched.')
    p.add_argument('--interface_res',
        help='File with space-separated integers of residue positions. Report rmsd on these '\
             'residues as "interface_rmsd"')
    p.add_argument('--outdir', type=str, help='Folder to output predicted structures')
    p.add_argument('--outcsv', type=str, help='Name of output csv file with metrics')
    p.add_argument('--trb_dir', type=str, help='Folder containing .trb files (if not same as pdb folder)')
    p.add_argument('--pdb_suffix', default='', help='PDB files have this suffix relative to trb files')
    p.add_argument('--model_num', default=4, type=int, choices=[1,2,3,4,5], help='AlphaFold model to use')
    p.add_argument('--use_ptm', default=False, action="store_true", help='Use ptm model variant')
    p.add_argument('--num_recycle', default=3, type=int, help='Number of recycles for AlphaFold prediction')
    p.add_argument('--amber_relax', action="store_true", help='Do AMBER relax after AF2 prediction.')


    if argv is not None:
        args = p.parse_args(argv)
    else:
        args = p.parse_args()

    if os.path.isdir(args.input_data):
        args.data_dir = args.input_data
    else:
        args.data_dir = os.path.dirname(args.input_data)

    if args.outcsv is None:
        args.outcsv = os.path.join(args.data_dir,'af2_metrics.csv')
    if args.outdir is None:
        args.outdir = os.path.join(args.data_dir,'af2/')
    os.makedirs(args.outdir, exist_ok=True)

    return args

def make_fixed_size(protein, shape_schema, msa_cluster_size, extra_msa_size,
                   num_res, num_templates=0):
  """Guess at the MSA and sequence dimensions to make fixed size."""
  NUM_RES = shape_placeholders.NUM_RES
  NUM_MSA_SEQ = shape_placeholders.NUM_MSA_SEQ                                                             
  NUM_EXTRA_SEQ = shape_placeholders.NUM_EXTRA_SEQ
  NUM_TEMPLATES = shape_placeholders.NUM_TEMPLATES

  pad_size_map = {
      NUM_RES: num_res,
      NUM_MSA_SEQ: msa_cluster_size,
      NUM_EXTRA_SEQ: extra_msa_size,
      NUM_TEMPLATES: num_templates,
  }

  for k, v in protein.items():
    # Don't transfer this to the accelerator.
    if k == 'extra_cluster_assignment':
      continue
    shape = list(v.shape)
    schema = shape_schema[k]
    assert len(shape) == len(schema), (
        f'Rank mismatch between shape and shape schema for {k}: '
        f'{shape} vs {schema}')
    pad_size = [
        pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)
    ]
    padding = [(0, p - tf.shape(v)[i]) for i, p in enumerate(pad_size)]
    if padding:
      protein[k] = tf.pad(
          v, padding, name=f'pad_to_fixed_{k}')
      protein[k].set_shape(pad_size)
  return {k:np.asarray(v) for k,v in protein.items()}

def idx2contigstr(idx):
    istart = 0
    contigs = []
    for iend in np.where(np.diff(idx)!=1)[0]:
            contigs += [f'{idx[istart]}-{idx[iend]}']
            istart = iend+1
    contigs += [f'{idx[istart]}-{idx[-1]}']
    return contigs

def calc_rmsd(xyz1, xyz2, eps=1e-6):

    # center to CA centroid
    xyz1 = xyz1 - xyz1.mean(0)
    xyz2 = xyz2 - xyz2.mean(0)

    # Computation of the covariance matrix
    C = xyz2.T @ xyz1

    # Compute optimal rotation matrix using SVD
    V, S, W = np.linalg.svd(C)

    # get sign to ensure right-handedness
    d = np.ones([3,3])
    d[:,-1] = np.sign(np.linalg.det(V)*np.linalg.det(W))

    # Rotation matrix U
    U = (d*V) @ W

    # Rotate xyz2
    xyz2_ = xyz2 @ U

    L = xyz2_.shape[0]
    rmsd = np.sqrt(np.sum((xyz2_-xyz1)*(xyz2_-xyz1), axis=(0,1)) / L + eps)

    return rmsd

def extract_contig_str(trb):
    if trb['settings']['contigs'] is not None:
        return trb['settings']['contigs']
    elif 'mask' in trb['settings']:
        return ','.join([x for x in trb['settings']['mask'].split(',') if x[0].isalpha()])

def contigs2idx(contigs):
    idx = []
    for con in contigs:
        idx.extend(np.arange(con[0],con[1]+1))
    return idx

def parse_range(_range):
    if '-' in _range:
      s, e = _range.split('-')
    else:
      s, e = _range, _range

    return int(s), int(e)

def parse_contig(contig):
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
        s, e = parse_range(contig[1:])
      else:
        s, e = None, None
    # is gap
    else:
      ch = None
      s, e = parse_range(contig)

    return ch, s, e

def expand(mask_str):
    '''
    Ex: '2,A3-5,3' --> [None, None, (A,3), (A,4), (A,5), None, None, None]
    '''
    expanded = []
    for l in mask_str.split(','):
      ch, s, e = parse_contig(l)

      # contig
      if ch:
        expanded += [(ch, res) for res in range(s, e+1)]
      # gap
      else:
        expanded += [None for _ in range(s)]

    return expanded

def main():

    args = get_args()

    # setup AF2 model
    model_name = f'model_{args.model_num}'
    if args.use_ptm:
        model_name += '_ptm'
    print(f'Using {model_name}')
    model_config = config.model_config(model_name)
    model_config.data.eval.num_ensemble = 1

    model_config.model.num_recycle = args.num_recycle
    model_config.data.common.num_recycle = args.num_recycle

    model_config.data.common.max_extra_msa = 1
    model_config.data.eval.max_msa_clusters = 1

    model_params = data.get_model_haiku_params(model_name=model_name, data_dir="/software/mlfold/alphafold-data")
    model_runner = model.RunModel(model_config, model_params)

    eval_cfg = model_config.data.eval
    model2crop_feats = {k:[None]+v for k,v in dict(eval_cfg.feat).items()}

    # inputs & outputs
    if args.template is not None:
        pdb_ref = parse_pdb(args.template)
        xyz_ref = pdb_ref['xyz'][:,:3]

    if args.interface_res is not None:
        interface_res = expand(args.interface_res)
        L_int = len(interface_res)

    # load all sequences for prediction
    seqs = []
    names = []
    if os.path.isdir(args.input_data):
        filenames = sorted(glob.glob(os.path.join(args.input_data,'*.pdb')))
    else:
        filenames = [args.input_data]

    for fn in filenames:
        seq = ''.join([util.aa_N_1[a] for a in parse_pdb(fn)['seq']])
        seqs.append(seq)
        names.append(os.path.basename(fn).replace('.pdb',''))

    # max length for padding
    Lmax = max([len(s) for s in seqs])

    print(f'{"name":>12}{"time (s)":>12}{"af2_lddt":>12}{"rmsd_af2_des":>18}{"contig_rmsd_af2":>18}',end='')

    if args.trb_dir is not None: trb_dir = args.trb_dir
    else: trb_dir = args.data_dir

    if args.interface_res is not None:
        print(f'{"interface_rmsd_af2":>20}')
    else:
        print()

    records = []
    for name, seq in zip(names, seqs):
        t0 = time.time()
        
        row = OrderedDict()
        row['name'] = name
        L = len(seq)

        if os.path.exists(os.path.join(args.outdir,name+'.pdb')) and \
           os.path.exists(os.path.join(args.outdir,name+'.npz')):

            print(f'Output already exists for {name}. Skipping AF2 prediction and calculating '\
                   'RMSD from existing pdb.')
            pdb_af2 = parse_pdb(os.path.join(args.outdir, name+'.pdb'))
            xyz_pred = pdb_af2['xyz'][:,:3] # BB atoms

            npz = np.load(os.path.join(args.outdir, name+'.npz'))
            row['af2_lddt'] = np.mean(npz['plddt'][:L])

        else:
            # run AF2
            feature_dict = {
                **pipeline.make_sequence_features(sequence=seq,description="none",num_res=len(seq)),
                **pipeline.make_msa_features(msas=[[seq]],deletion_matrices=[[[0]*len(seq)]]),
            }
            inputs = model_runner.process_features(feature_dict, random_seed=0)
            inputs_padded = make_fixed_size(inputs, model2crop_feats, msa_cluster_size=0, extra_msa_size=0, num_res=Lmax, num_templates=0)

            outputs = model_runner.predict(inputs_padded)
            xyz_pred = np.array(outputs['structure_module']['final_atom_positions'][:L,:3]) # N, Ca, C
            row['af2_lddt'] = np.mean(outputs['plddt'][:L])
            
            # unrelaxed_protein = protein.from_prediction(inputs_padded,outputs)
            unrelaxed_protein = protein.Protein(
                aatype=inputs_padded['aatype'][0][:L],
                atom_positions=outputs['structure_module']['final_atom_positions'][:L],
                atom_mask=outputs['structure_module']['final_atom_mask'][:L],
                residue_index=inputs_padded['residue_index'][0][:L]+1,
                b_factors=np.zeros_like(outputs['structure_module']['final_atom_mask'])
            )
            pdb_lines = protein.to_pdb(unrelaxed_protein)

            # relax, if wanted
            if args.amber_relax:
                amber_relaxer = relax.AmberRelaxation(
                    max_iterations=0,
                    tolerance=2.39,
                    stiffness=10.0,
                    exclude_residues=[],
                    max_outer_iterations=20
                )
                pdb_lines, _, _ = amber_relaxer.process(prot=unrelaxed_protein)

            # save AF2 pdb
            with open(os.path.join(args.outdir,name+'.pdb'), 'w') as f:
                f.write(pdb_lines) 

            # save AF2 residue-wise plddt, pAE, pTM
            np.savez(os.path.join(args.outdir, name+'.npz'),
                 plddt=outputs['plddt'][:L],
                 max_predicted_aligned_error=outputs.get('max_predicted_aligned_error'),
                 ptm=outputs.get('ptm')
            )

        # load designed pdb
        pdb_des = parse_pdb(os.path.join(args.data_dir, name+'.pdb'))
        xyz_des = pdb_des['xyz'][:,:3] # extract N,CA,C coords

        # run metadata
        trbname = os.path.join(trb_dir, name+args.pdb_suffix+'.trb')
        if os.path.exists(trbname): 
            trb = np.load(trbname,allow_pickle=True)

        # load reference structure, if needed
        if args.template is None and args.template_dir is None and os.path.exists(trbname):
            pdb_ref = parse_pdb(trb['settings']['pdb'])
            xyz_ref = pdb_ref['xyz'][:,:3]
        if args.template_dir is not None and os.path.exists(trbname):
            pdb_ref = parse_pdb(args.template_dir+trb['settings']['pdb'].split('/')[-1])
            xyz_ref = pdb_ref['xyz'][:,:3]

        # recalculate 0-indexed motif residue positions (sometimes they're wrong, e.g. from inpainting)
        if os.path.exists(trbname):
        #if os.path.exists(trbname) and 'con_ref_idx0' not in trb:
            idxmap = dict(zip(pdb_ref['pdb_idx'],range(len(pdb_ref['pdb_idx']))))
            trb['con_ref_idx0'] = np.array([idxmap[i] for i in trb['con_ref_pdb_idx']])
            idxmap = dict(zip(pdb_des['pdb_idx'],range(len(pdb_des['pdb_idx']))))
            trb['con_hal_idx0'] = np.array([idxmap[i] for i in trb['con_hal_pdb_idx']])

        # calculate rmsds
        row['rmsd_af2_des'] = calc_rmsd(xyz_pred.reshape(L*3,3), xyz_des.reshape(L*3,3))
 
        # load contig position
        if os.path.exists(trbname): 
            idx_motif = [i for i,idx in zip(trb['con_hal_idx0'],trb['con_ref_pdb_idx']) 
                         if idx[0]!='R']
            # TODO: renumber if design has been trimmed for fastdesign

            L_motif = len(idx_motif)

            idx_motif_ref = [i for i,idx in zip(trb['con_ref_idx0'],trb['con_ref_pdb_idx']) 
                             if idx[0]!='R']
            xyz_ref_motif = xyz_ref[idx_motif_ref]
            row['contig_rmsd_af2_des'] = calc_rmsd(xyz_pred[idx_motif].reshape(L_motif*3,3), 
                                                   xyz_des[idx_motif].reshape(L_motif*3,3))
            row['contig_rmsd_af2'] = calc_rmsd(xyz_pred[idx_motif].reshape(L_motif*3,3), xyz_ref_motif.reshape(L_motif*3,3))

            if args.interface_res is not None: 
                idxmap = dict(zip(trb['con_ref_pdb_idx'],trb['con_ref_idx0']))
                idxmap2 = dict(zip(trb['con_ref_pdb_idx'],trb['con_hal_idx0']))
                idx_int_ref = [idxmap[i] for i in interface_res if i in trb['con_ref_pdb_idx']]
                idx_int_hal = [idxmap2[i] for i in interface_res if i in trb['con_ref_pdb_idx']]
                L_int = len(idx_int_ref)

                row['interface_rmsd_af2'] = calc_rmsd(xyz_pred[idx_int_hal].reshape(L_int*3,3), xyz_ref[idx_int_ref].reshape(L_int*3,3))
                row['interface_rmsd_af2_des'] = calc_rmsd(xyz_pred[idx_int_hal].reshape(L_int*3,3), xyz_des[idx_int_hal].reshape(L_int*3,3))

        records.append(row)

        t = time.time() - t0
        print(f'{name:>12}{t:>12.2f}{row["af2_lddt"]:>12.2f}{row["rmsd_af2_des"]:>18.2f}',end='')
        if os.path.exists(trbname):
            print(f'{row["contig_rmsd_af2"]:>18.2f}',end='')
        if args.interface_res is not None:
            print(f'{row["interface_rmsd_af2"]:>20.2f}',end='')
        print()

    df = pd.DataFrame.from_records(records)

    print(f'Outputting computed metrics to {args.outcsv}')
    df.to_csv(args.outcsv)

if __name__ == "__main__":
    main()
