import sys, os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from parsers import parse_a3m, read_templates
from TrunkModel  import TrunkModule
import util
from collections import namedtuple
from ffindex import *
from kinematics import xyz_to_c6d, c6d_to_bins2, xyz_to_t2d
from trFold import TRFold

NBIN = [37, 37, 37, 19]

MODEL_PARAM ={
        "n_module"     : 8,
        "n_module_str" : 4,
        "n_layer"      : 1,
        "d_msa"        : 384 ,
        "d_pair"       : 288,
        "d_templ"      : 64,
        "n_head_msa"   : 12,
        "n_head_pair"  : 8,
        "n_head_templ" : 4,
        "d_hidden"     : 64,
        "r_ff"         : 4,
        "n_resblock"   : 1,
        "p_drop"       : 0.0,
        "use_templ"    : True,
        "performer_N_opts": {"nb_features": 64},
        "performer_L_opts": {"nb_features": 64}
        }

SE3_param = {
        "num_layers"    : 3,
        "num_channels"  : 16,
        "num_degrees"   : 2,
        "l0_in_features": 32,
        "l0_out_features": 8,
        "l1_in_features": 3,
        "l1_out_features": 2,
        "num_edge_features": 32,
        "div": 2,
        "n_heads": 4
        }
MODEL_PARAM['SE3_param'] = SE3_param

# params for the folding protocol
fold_params = {
    "SG7"     : np.array([[[-2,3,6,7,6,3,-2]]])/21,
    "SG9"     : np.array([[[-21,14,39,54,59,54,39,14,-21]]])/231,
    "DCUT"    : 19.5,
    "ALPHA"   : 1.57,
    
    # TODO: add Cb to the motif
    "NCAC"    : np.array([[-0.676, -1.294,  0.   ],
                          [ 0.   ,  0.   ,  0.   ],
                          [ 1.5  , -0.174,  0.   ]], dtype=np.float32),
    "CLASH"   : 2.0,
    "PCUT"    : 0.5,
    "DSTEP"   : 0.5,
    "ASTEP"   : np.deg2rad(10.0),
    "XYZRAD"  : 7.5,
    "WANG"    : 0.1,
    "WCST"    : 0.1
}

fold_params["SG"] = fold_params["SG9"]

class Predictor():
    def __init__(self, model_dir=None, device="cuda:0"):
        if model_dir == None:
            self.model_dir = "%s/models"%(os.path.dirname(os.path.realpath(__file__)))
        else:
            self.model_dir = model_dir
        #
        # define model name
        self.model_name = "BFF"
        self.device = device
        self.active_fn = nn.Softmax(dim=1)

        # define model & load model
        self.model = TrunkModule(**MODEL_PARAM).to(self.device)
        could_load = self.load_model(self.model_name)
        if not could_load:
            print ("ERROR: failed to load model")
            sys.exit()

    def load_model(self, model_name, suffix='last'):
        chk_fn = "%s/%s_%s.pt"%(self.model_dir, model_name, suffix)
        print (chk_fn)
        if not os.path.exists(chk_fn):
            return False
        checkpoint = torch.load(chk_fn, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return True
    
    def predict(self, a3m_fn, out_prefix, hhr_fn=None, atab_fn=None, window=150, shift=50, n_latent=128):
        msa_orig = parse_a3m(a3m_fn)
        N, L = msa_orig.shape
        #
        if os.path.exists(hhr_fn):
            xyz_t, t1d, t0d = read_templates(L, ffdb, hhr_fn, atab_fn, n_templ=10)
        else:
            xyz_t = torch.full((1, L, 3, 3), np.nan).float()
            t1d = torch.zeros((1, L, 1)).float()
            t0d = torch.zeros((1,3)).float()
        #
        # template features
        xyz_t = xyz_t.float().unsqueeze(0)
        t1d = t1d.float().unsqueeze(0)
        t0d = t0d.float().unsqueeze(0)
        t2d = xyz_to_t2d(xyz_t, t0d)
       
        self.model.eval()
        for i_trial in range(10):
            self.run_prediction(msa_orig, t1d, t2d, "%s_%02d"%(out_prefix, i_trial), n_latent=n_latent)
            torch.cuda.empty_cache()
    def run_prediction(self, msa_orig, t1d, t2d, out_prefix, n_latent=128):
        N, L = msa_orig.shape
        with torch.no_grad():
            #
            msa = torch.tensor(msa_orig).long() # (N, L)
            random_idx = torch.randperm(N-1)
            msa = torch.cat([msa[:1,:], msa[1:,:][random_idx]], dim=0).view(1, N, L)
            #
            if n_latent < N:
                msa_latent = msa[:,:n_latent]
                msa_extra = msa[:,n_latent:,:]
            else:
                msa_latent = msa
                msa_extra = msa[:,:1,:]
            msa_extra = msa_extra[:,:10000]
            print (msa_latent.shape, msa_extra.shape, N, L)
            #
            idx_pdb = torch.arange(L).long().view(1, L)
            #
            msa_latent = msa_latent.to(self.device)
            msa_extra = msa_extra.to(self.device)
            seq = msa_latent[:,0]
            idx_pdb = idx_pdb.to(self.device)
            t1d = t1d.to(self.device)
            t2d = t2d.to(self.device)
            with torch.cuda.amp.autocast(enabled=True):
                logit_s, init_crds, pred_lddt = self.model(msa_latent, msa_extra, seq, idx_pdb, t1d=t1d, t2d=t2d)
            prob_s = list()
            for logit in logit_s:
                prob = self.active_fn(logit.float()) # distogram
                prob = prob.reshape(-1, L, L) #.permute(1,2,0).cpu().numpy()
                prob_s.append(prob)
        
        for prob in prob_s:
            prob += 1e-8
            prob = prob / torch.sum(prob, dim=0)[None]
        self.write_pdb(seq[0], init_crds[0], Bfacts=pred_lddt[0], prefix="%s_init"%(out_prefix))
        xyz = init_crds[0, :, 1] # initial ca coordinates
        TRF = TRFold(prob_s, fold_params)
        xyz = TRF.fold(xyz, batch=45, lr=0.1, nsteps=200)
        self.write_pdb(seq[0], xyz, prefix="%s"%(out_prefix), Bfacts=pred_lddt[0])

        prob_s = [prob.permute(1,2,0).detach().cpu().numpy().astype(np.float16) for prob in prob_s]
        np.savez_compressed("%s.npz"%(out_prefix), dist=prob_s[0].astype(np.float16), \
                            omega=prob_s[1].astype(np.float16),\
                            theta=prob_s[2].astype(np.float16),\
                            phi=prob_s[3].astype(np.float16))

                    
    def write_pdb(self, seq, atoms, Bfacts=None, prefix=None):
        L = len(seq)
        filename = "%s.pdb"%prefix
        ctr = 1
        with open(filename, 'wt') as f:
            if Bfacts == None:
                Bfacts = np.zeros(L)
            else:
                Bfacts = torch.clamp( Bfacts, 0, 1)
            
            for i,s in enumerate(seq):
                if (len(atoms.shape)==2):
                    f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                            "ATOM", ctr, " CA ", util.num2aa[s], 
                            "A", i+1, atoms[i,0], atoms[i,1], atoms[i,2],
                            1.0, Bfacts[i] ) )
                    ctr += 1

                elif atoms.shape[1]==3:
                    for j,atm_j in enumerate((" N  "," CA "," C  ")):
                        f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                                "ATOM", ctr, atm_j, util.num2aa[s], 
                                "A", i+1, atoms[i,j,0], atoms[i,j,1], atoms[i,j,2],
                                1.0, Bfacts[i] ) )
                        ctr += 1                
        
def get_args():
    #DB="/home/robetta/rosetta_server_beta/external/databases/trRosetta/pdb100_2021Mar03/pdb100_2021Mar03"
    DB = "/projects/ml/TrRosetta/pdb100_2020Mar11/pdb100_2020Mar11"
    import argparse
    parser = argparse.ArgumentParser(description="RoseTTAFold: Protein structure prediction with 3-track attentions on 1D, 2D, and 3D features")
    parser.add_argument("a3m_fn", help="input MSA in a3m format")
    parser.add_argument("out_prefix", help="prefix for output file. [out_prefix].npz file will be generated")
    parser.add_argument("hhr_fn",
                        help="HHsearch result file in hhr format")
    parser.add_argument("atab_fn",
                        help="HHsearch result file in atab format")
    parser.add_argument("-db", default=DB, required=False, 
                        help="HHsearch database [%s]"%DB)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if not os.path.exists("%s.npz"%args.out_prefix):
        FFDB = args.db
        FFindexDB = namedtuple("FFindexDB", "index, data")
        ffdb = FFindexDB(read_index(FFDB+'_pdb.ffindex'),
                         read_data(FFDB+'_pdb.ffdata'))
        pred = Predictor()
        pred.predict(args.a3m_fn, args.out_prefix, args.hhr_fn, args.atab_fn)
