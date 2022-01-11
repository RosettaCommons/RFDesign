import sys, os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from functools import partial
import argparse 

# from minkyung 
from data_loader import get_train_valid_set,\
                        loader_tbm, Dataset,\
                        tbm_collate_fn, loader_fixbb,\
                        collate_fixbb, loader_mix, collate_mix,\
                        DeNovoDataset, loader_denovo, collate_denovo

from kinematics import xyz_to_c6d, c6d_to_bins2, xyz_to_t2d, xyz_to_bbtor
from TrunkModel  import TrunkModule
from loss import *
import util 
from scheduler import get_linear_schedule_with_warmup, CosineAnnealingWarmupRestarts

# dj external functions 
import dj_util

from icecream import ic 

os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # debbugging


torch.set_printoptions(sci_mode=False)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    
    # model params from Minkyung's pretrained checkpoint 

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--checkpoint', '-ckpt',    type=str,   default='/home/davidcj/projects/BFF/design/dj_design/from_minkyung_072021/models/BFF_last.pt',  help='Model checkpoint to load')
    parser.add_argument('--outf',       '-outf',   type=str,   default='./pdbs_test',                  help='directory to spill output into')
    parser.add_argument('--lmin',       '-lmin',    type=int,   default=300,                            help='minimum chain length through model')
    parser.add_argument('--lmax',       '-lmax',    type=int,   default=301,                            help='max chain length through model')
    parser.add_argument('--dump_pdb',   '-dump_pdb',            default=False,  action='store_true',    help='dump pdb files into output directories?')

    args = parser.parse_args()

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    

    # parameters for loader 
    loader_param = {
        'DATCUT': '2030-Jan-01',
        'DIR': '/projects/ml/TrRosetta/PDB30-20FEB17',
        'LIST': '/projects/ml/TrRosetta/PDB30-20FEB17/list.no_denovo.csv',
        'LMAX': args.lmax,
        'LMIN': args.lmin,
        'MAXLAT': 50,
        'MAXSEQ': 1000,
        'MAXTOKEN': 65536,
        'MAXTPLT': 10,
        'MINSEQ': 1,
        'MINTPLT': 0,
        'RESCUT': 3.5,
        'ROWS': 1,
        'SLICE': 'CONT',
        'SUBSMP': 'LOG',
        'VAL': '/projects/ml/TrRosetta/PDB30-20FEB17/val_lists/xaa',
        'seqID': 150.0
        }

    
    # model params from Minkyung's pretrained checkpoint 

    model_param = {'SE3_param': {'div': 2,
                                'l0_in_features': 32,
                                'l0_out_features': 8,
                                'l1_in_features': 3,
                                'l1_out_features': 2,
                                'n_heads': 4,
                                'num_channels': 16,
                                'num_degrees': 2,
                                'num_edge_features': 32,
                                'num_layers': 3},
                  'd_hidden': 64,
                  'd_msa': 384,
                  'd_msa_full': 64,
                  'd_pair': 288,
                  'd_templ': 64,
                  'n_head_msa': 12,
                  'n_head_pair': 8,
                  'n_head_templ': 4,
                  'n_layer': 1,
                  'n_module': 8,
                  'n_module_str': 4,
                  'n_resblock': 1,
                  'p_drop': 0.15,
                  'performer_L_opts': {'feature_redraw_interval': 10000, 'nb_features': 64},
                  'performer_N_opts': {'feature_redraw_interval': 10000, 'nb_features': 64},
                  'r_ff': 4}
    model_param['use_templ'] = True


    # paths to denovo proteins 
    denovo_list1 = '/projects/ml/TrRosetta/benchmarks/denovo2021/denovo.txt'
    denovo_list2 = '/projects/ml/TrRosetta/benchmarks/denovo2021/update_19Jul2021.txt'

    # folders containing denovo xtals in .pdb files and .fa files 
    denovo_pdb = '/projects/ml/TrRosetta/benchmarks/denovo2021/pdb/'
    denovo_fas = '/projects/ml/TrRosetta/benchmarks/denovo2021/fas/'
    
    # Make denovo datasets / loaders 
    dataset1 = DeNovoDataset(denovo_list1, denovo_pdb, denovo_fas, loader_denovo)
    dataset2 = DeNovoDataset(denovo_list2, denovo_pdb, denovo_fas, loader_denovo)

    dataloader1 = data.DataLoader(dataset1, batch_size=1, collate_fn=partial(collate_denovo, params=loader_param))
    dataloader2 = data.DataLoader(dataset2, batch_size=1, collate_fn=partial(collate_denovo, params=loader_param))
    

    # make model 
    model = TrunkModule(**model_param).to(DEVICE)


    # load checkpoint into model 
    print('Loading checkpoint')
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    print('Successfully loaded pretrained weights into model')

    # cross entropy loss 
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    full_loss_s = []
    names = []
    tasks = []
    ctr = 0

    # Loop through proteins and make predictions
    with torch.no_grad():
        model.eval()
        for msa, msa_full, true_crds, idx_pdb, xyz_t, t1d, t0d, name in dataloader1:
            ctr += 1

            for _iter in range(2): # two iters: (1) seq --> str, (2) str --> seq 

                loss_s = []
                names += name

                
                label_aa_s = torch.clone(msa).to(DEVICE)

                if _iter == 0:
                    # structure prediction task 
                    task = 'str' 
                    mask_aa_s = torch.full_like(msa, False).bool()
                else:
                    # sequence prediction task 
                    task = 'seq'
                    
                    # mask out the sequence information 
                    msa       = torch.full_like(msa, 21)
                    msa_full  = torch.clone(msa) 
                    mask_aa_s = torch.full_like(msa, True).bool()

                    xyz_t = torch.clone(true_crds)[None,...]
                    t1d = torch.ones_like(t1d)
                    t0d = torch.ones_like(t0d)
                

                # keep track of which task we are doing 
                tasks.append(task)


                msa       = msa.to(DEVICE).long()
                msa_full  = msa_full.to(DEVICE).long()
                idx_pdb   = idx_pdb.to(DEVICE).long()
                true_crds = true_crds.to(DEVICE).float()


                c6d, _ = xyz_to_c6d(true_crds)
                c6d    = c6d_to_bins2(c6d)
                tors   = xyz_to_bbtor(true_crds)

                xyz_t = xyz_t.to(DEVICE).float()
                t1d   = t1d.to(DEVICE).float()
                t0d   = t0d.to(DEVICE).float()

                print('Before t2d')
                ic(xyz_t.shape)
                ic(t0d.shape)
                t2d   = xyz_to_t2d(xyz_t, t0d)
                
                if task == 'str':
                    seq = msa[:,0]
                else:
                    seq = torch.full_like(msa[:,0], 20)
                
                # forward pass through model 
                ic(msa.shape)
                ic(msa_full.shape)
                ic(seq.shape)
                ic(idx_pdb.shape)
                ic(t1d.shape)
                ic(t2d.shape)
                logit_s, logit_aa_s, logit_tor_s, pred_crds, pred_lddts = model(msa, msa_full, seq, idx_pdb,
                                                                                t1d=t1d, t2d=t2d,
                                                                                use_transf_checkpoint=True)
                
                if args.dump_pdb:
                    util.writepdb(os.path.join(args.outf, name[0]) + '_pred.pdb', pred_crds[-1].squeeze(), torch.ones(pred_crds.shape[2]), seq.squeeze())


                
                # NAMES GOING INTO CALC LOSS
                # logit_s, c6d,
                # logit_aa_s, msa, mask,
                # logit_tor_s, tors,
                # pred_crds, true_crds, 
                # pred_lddts, idx_pdb 

                # NAMES INSIDE CALC LOSS 
                # logit_s, label_s,
                # logit_aa_s, label_aa_s, mask_aa_s
                # logit_tor_s, label_tor_s,
                # pred, true, 
                # pred_lddt, idx

                    
                ##### calculate various losses ##### 
                B,L = true_crds.shape[:2]


                # C6d Losses 
                for i in range(4):
                    loss = loss_fn(logit_s[i], c6d[...,i]).mean()
                    loss_s.append(loss[None].detach().item())

                # masked token prediction loss 

                loss = loss_fn(logit_aa_s, label_aa_s.reshape(B, -1))
                loss = loss.mean() # just take the mean because it's over the whole sequence 
                loss_s.append(loss[None].detach().item())

                # masked token prediction accuracy 
                TOP1,TOP2,TOP3 = 1,2,3
                logits_argsort = torch.argsort(logit_aa_s, dim=1, descending=True)           # sort down the dim of 21
                
                #ic(logits_argsort.shape)
                #ic(label_aa_s[...,mask_aa_s])
                #ic(mask_aa_s)
                #ic(logits_argsort[:,:TOP1, mask_aa_s.reshape(B,-1).squeeze()])
                #ic(logits_argsort[:,:TOP2, mask_aa_s.reshape(B,-1).squeeze()])
                acc1 = (label_aa_s[...,mask_aa_s] == logits_argsort[:, :TOP1, mask_aa_s.reshape(B,-1).squeeze()]).any(dim=1).float().mean()[None] # top-1 average accuracy
                acc2 = (label_aa_s[...,mask_aa_s] == logits_argsort[:, :TOP2, mask_aa_s.reshape(B,-1).squeeze()]).any(dim=1).float().mean()[None] # top-2 average accuracy
                acc3 = (label_aa_s[...,mask_aa_s] == logits_argsort[:, :TOP3, mask_aa_s.reshape(B,-1).squeeze()]).any(dim=1).float().mean()[None] # top-3 average accuracy

                #ic(acc1, acc2, acc3)

                loss_s.extend([acc1.detach().item(), acc2.detach().item(), acc3.detach().item()]) # append accuracies to loss list for recording
                
                # c6d losses 
                for i in range(4):
                    loss = loss_fn(logit_s[i], c6d[...,i]).mean()
                    loss_s.append(loss[None].detach().item())

                # torsion angle loss 
                for i in range(2):
                    loss = loss_fn(logit_tor_s[i], tors[...,i]).mean()
                    loss_s.append(loss[None].detach().item())


                # ca_lddt loss 
                _, ca_lddt = calc_lddt_loss(pred_crds[:,:,:,1], true_crds[:,:,1], pred_lddts, idx_pdb)
                loss_s.extend([a.item() for a in ca_lddt.detach()])

                # bond geometry 
                blen_loss, bang_loss = calc_BB_bond_geom(pred_crds[-1:], true_crds)
                loss_s.extend([a.item() for a in torch.stack((blen_loss, bang_loss)).detach()])
                
                full_loss_s.append(loss_s)
        
        # finished going through entire loader - save the data 
        with open(os.path.join(args.outf, 'denovo_pred.txt'), 'w') as F:
            
            F.write(' '.join(['name','task','c6d_1','c6d_2', 'c6d_3', 'c6d_4', 'aa_ce','acc1','acc2','acc3','c6d1', 'c6d2', 'c6d3', 'c6d4', 'tors1', 'tors2',\
                'ca_lddt1', 'ca_lddt2', 'ca_lddt3', 'ca_lddt4', 'ca_lddt5', 'ca_lddt6',\
                'blen', 'bang']) + '\n')
            for i,row in enumerate(full_loss_s):

                name = names[i]
                task = tasks[i]

                line = ' '.join([name,task] + [f'{a:.4f}' for a in row]) + '\n'

                F.write(line)


if __name__ == '__main__':
    main()
