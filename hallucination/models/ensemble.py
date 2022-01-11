import torch
import torch.nn as nn

class EnsembleNet(nn.Module):
    ''''''
    def __init__(self, Net, params, checkpoints):
        super(EnsembleNet, self).__init__()

        nets = []
        print('# ' + '-'*78)
        for name in checkpoints:
            print('# loading %s . . .'%(name))
            chk = torch.load(name)
            if 'model_state_dict' in chk.keys():
                chk = chk['model_state_dict']
            net = Net(**params)
            net.load_state_dict(chk, strict=False)
            nets.append(net)

        self.nets = nn.ModuleList(nets)
        
    def forward(self,msa,idx=None, msa_one_hot=None):
        
        # run all networks in the ensemble
        if msa_one_hot != None:
            preds = [net(msa,idx,msa_one_hot=msa_one_hot) for net in self.nets]
        else:
            preds = [net(msa,idx) for net in self.nets]
        
        # average predictions over the ensemble
        out = {key:torch.stack([p[key] for p in preds],dim=-1).mean(dim=-1) 
               for key in preds[0].keys()}
        
        return out
