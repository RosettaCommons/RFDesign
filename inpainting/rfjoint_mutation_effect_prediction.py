import os, sys
import time
import numpy as np
import torch
import torch.nn as nn
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir+'/model')
from RoseTTAFoldModel  import RoseTTAFoldModule
import util
from kinematics import xyz_to_c6d,c6d_to_bins, c6d_to_bins2, xyz_to_t2d, xyz_to_bbtor
from parsers import parse_a3m, parse_pdb_mutation_effect

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

class Predictor():
    def __init__(self, use_cpu=False, checkpoint=None,params={'MAXLAT': 256, 'MAXSEQ': 1024}):
        if torch.cuda.is_available() and (not use_cpu):
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
        self.params = params
        
        # define model & load model
        self.model = RoseTTAFoldModule(**MODEL_PARAM).to(self.device)
        ckpt = torch.load(checkpoint,map_location=self.device)
        model_state = ckpt['model_state_dict']
        self.model.load_state_dict(model_state)
        print('Successfully loaded model checkpoint')
        
        # freeze all parameters of rosettafold model
        for p in self.model.parameters():
            p.requires_grad = False

    def predict(self,a3m_fn,out_dir,out_name='rfjoint_mut_pred',templ_fn=None):
        # msa processing 
        msa_orig = parse_a3m(a3m_fn)['msa']
        if (len(msa_orig) > self.params['MAXLAT']):
            msa_start = msa_orig[:self.params['MAXLAT']]
            msa_end = msa_orig[self.params['MAXLAT']:self.params['MAXLAT']+self.params['MAXSEQ']]
        if (len(msa_orig) <= self.params['MAXLAT']): 
            msa_start = msa_orig[:self.params['MAXLAT']]
            msa_end = msa_orig[:self.params['MAXLAT']]

        msa_start = torch.Tensor(msa_start)
        msa_end = torch.Tensor(msa_end)

        if templ_fn: 
            #if template provided
            pdb_orig,idx_orig = parse_pdb_mutation_effect(templ_fn)
            pdb_orig,idx_orig= torch.Tensor(pdb_orig[None,:,:,:]).permute(0,2,1,3) ,torch.tensor(idx_orig[None,:])
            
        else: 
            #idx in this case is just the np.arange 
            idx_orig = torch.Tensor(np.arange(1,msa_start.shape[-1]+1))[None,]
            #no template setting - 0s for 1d and 0d features and NaN for structure 
            pdb_orig = np.zeros((1,msa_start.shape[-1],3,3))
            pdb_orig[pdb_orig == 0] = np.nan
            pdb_orig = torch.Tensor(pdb_orig)
        
        
        zero_shot_prediction = np.zeros((msa_start.shape[1],21))
        with torch.no_grad():
            self.model.eval()
            for i in range(msa_start.shape[1]): # go through every amino acid
                N , L = msa_start.shape[0], msa_start.shape[1]
                msa_mut = msa_start.clone()
                msa_mut[0,i] = 21
                msa_start_onehot = torch.nn.functional.one_hot(msa_mut.clone().to(torch.int64), num_classes=22)
                fake_ins = torch.zeros_like(msa_start_onehot)[:,:,:2]
                msa_start_onehot_cat = torch.cat([msa_start_onehot, msa_start_onehot, fake_ins], dim=-1)
                msa_end_onehot = torch.nn.functional.one_hot(torch.Tensor(msa_end).clone().to(torch.int64), num_classes=22)
                fake_ins = torch.zeros_like(msa_end_onehot)[:,:,:1]
                msa_end_onehot_cat = torch.cat([msa_end_onehot ,fake_ins], dim=-1)[None]
                raw_seq = msa_start[0][None]
                mut_seq = raw_seq.clone().long()
                mut_seq[0,i] = 20 # mask out the sequence too 
                seq_onehot = torch.nn.functional.one_hot(mut_seq.to(torch.int64), num_classes=21) # one hot sequence 
                conf1d = torch.ones_like(torch.Tensor(msa_orig)[0][None][...,None])
                t1d = torch.cat((seq_onehot, conf1d), dim=-1)
                t2d = xyz_to_t2d(pdb_orig[None])

                msa_hot = msa_start_onehot_cat.to(self.device).float().unsqueeze(0)
                msa_extra_hot = msa_end_onehot_cat.to(self.device).float()
                xyz_t   = pdb_orig.to(self.device).float()
                f1d_t   = t1d.to(self.device).float().unsqueeze(0)
                t2d     = t2d.to(self.device)
                idx_pdb = idx_orig.to(self.device)
                mut_seq = mut_seq.to(self.device)
                logits, logits_aa, xyz, lddt = self.model(msa_latent=msa_hot, 
                                                         msa_full=msa_extra_hot, 
                                                         seq=mut_seq, 
                                                         idx=idx_pdb, t1d=f1d_t, 
                                                         t2d=t2d,use_checkpoint=True)


                logit_aa_s = logits_aa.reshape(-1, N, L).permute(1,2,0)
                logit_aa_s = nn.LogSoftmax(dim=-1)(logit_aa_s) # log probabilities
                zero_shot_prediction[i] = logit_aa_s[0][i].cpu().detach()

        print('Done with mutation effect prediction, saving as: %s/%s'%(out_dir,out_name),flush=True) 
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        np.save(out_dir+'/'+out_name,zero_shot_prediction)
        
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="RFJoint for mutation effect prediction.")
    parser.add_argument("-msa", dest="a3m_fn", required=True,
                        help="File path for input MSA in a3m format")
    parser.add_argument("-out_dir", dest="out_dir", required=True,
                        help="Output directory. a numpy file will be created (Lx21) with probabilities of each amino acid")
    parser.add_argument("-out_file_name", dest="out_name", required=False,default='rfjoint_mut_pred',
                        help="name of output file")
    parser.add_argument("-templ", dest="templ_fn", required=False, default=None,
                        help="Template PDB file if you want to use [optional]. Confidence will be assigned to 1 (perfect). The residue numbers should be matched to the given sequence in MSA. (resNo 10 in the PDB = 10th residue at the given sequence")
    parser.add_argument("--cpu", dest='use_cpu', default=False, action='store_true', help="Force to use CPU instead of GPU [False]")
    parser.add_argument("-n_seed", default=256,
                        help="Number of seed sequences [256]")
    parser.add_argument("-n_extra", default=1024,
                        help="Number of extra sequences [1024]")
    parser.add_argument('--checkpoint',dest='checkpoint', default=script_dir+'/weights/BFF_mix_epoch25.pt',
            help='Checkpoint to inpainting model')
    args = parser.parse_args()
    print(args,flush=True)
    params = {}
    params['MAXLAT'] = int(args.n_seed)
    params['MAXSEQ'] = int(args.n_extra)
    return args, params

if __name__ == "__main__":
    args, params = get_args()
    pred = Predictor(use_cpu=args.use_cpu,checkpoint=args.checkpoint,params=params)
    pred.predict(args.a3m_fn,args.out_dir,args.out_name,args.templ_fn,)


