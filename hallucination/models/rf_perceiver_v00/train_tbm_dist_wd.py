import sys, os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from functools import partial
from data_loader import get_train_valid_set, loader_tbm, Dataset, tbm_collate_fn
from kinematics import xyz_to_c6d, c6d_to_bins2, xyz_to_t2d, xyz_to_bbtor
from TrunkModel  import TrunkModule
from loss import *
from scheduler import get_linear_schedule_with_warmup, CosineAnnealingWarmupRestarts

# distributed data parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
#torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
## To reproduce errors
#import random
#np.random.seed(0)
#random.seed(0)

torch.set_num_threads(4)
N_GPU = torch.cuda.device_count()

N_PRINT_TRAIN = 64
#BATCH_SIZE = 1 * torch.cuda.device_count()

LOAD_PARAM = {'shuffle': False,
              'num_workers': 3,
              'pin_memory': True}

def DDP_setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '%d'%port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def DDP_cleanup():
    dist.destroy_process_group()

def add_weight_decay(model, l2_coeff):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        #if len(param.shape) == 1 or name.endswith(".bias"):
        if "norm" in name or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_coeff}]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer():
    def __init__(self, model_name='BFF',
                 n_epoch=100, step_lr=100, lr=1.0e-4, l2_coeff=1.0e-2, port=None,
                 model_param={}, loader_param={}, loss_param={}, batch_size=1, accum_step=1):
        self.model_name = "BFF"
        #self.model_name = "%s_%d_%d_%d_%d"%(model_name, model_param['n_module'], 
        #                                    model_param['n_module_str'],
        #                                    model_param['d_msa'],
        #                                    model_param['d_pair'])
        #
        self.n_epoch = n_epoch
        self.step_lr = step_lr
        self.init_lr = lr
        self.l2_coeff = l2_coeff
        self.port = port
        #
        self.model_param = model_param
        self.model_param['use_templ'] = True
        self.loader_param = loader_param
        self.loss_param = loss_param
        self.ACCUM_STEP = accum_step
        self.batch_size = batch_size
        print (self.model_param)
        print (self.loader_param)
        print (self.loss_param)

        # loss & final activation function
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.active_fn = nn.Softmax(dim=1)
        
    def calc_loss(self, logit_s, label_s,
                  logit_aa_s, label_aa_s, mask_aa_s,
                  logit_tor_s, label_tor_s,
                  pred, true, pred_lddt, idx,
                  w_dist=1.0, w_aa=1.0, w_str=1.0, w_rms=1.0,
                  w_lddt=1.0, w_blen=1.0, w_bang=1.0):
        B, L = true.shape[:2]

        loss_s = list()
        tot_loss = 0.0
       
        # c6d loss
        for i in range(4):
            loss = self.loss_fn(logit_s[i], label_s[...,i]).mean()
            tot_loss += w_dist*loss
            loss_s.append(loss[None].detach())

        # masked token prediction loss
        loss = self.loss_fn(logit_aa_s, label_aa_s.reshape(B, -1))
        loss = loss * mask_aa_s.reshape(B, -1)
        loss = loss.sum() / mask_aa_s.sum()
        tot_loss += w_aa*loss
        loss_s.append(loss[None].detach())
        
        # torsion angle loss
        for i in range(2):
            loss = self.loss_fn(logit_tor_s[i], label_tor_s[...,i]).mean()
            tot_loss += w_dist*0.5*loss
            loss_s.append(loss[None].detach())
        
        # Structural loss
        tot_str, str_loss = calc_str_loss(pred, true)
        tot_loss += w_str*tot_str
        loss_s.append(str_loss)
        
        lddt_loss, ca_lddt = calc_lddt_loss(pred[:,:,:,1], true[:,:,1], pred_lddt, idx)
        tot_loss += w_str*w_lddt*lddt_loss
        loss_s.append(lddt_loss.detach()[None])
        loss_s.append(ca_lddt.detach())
        
        dih_loss = calc_pseudo_dih(pred[:,:,:,1], true[:,:,1])
        tot_loss += dih_loss

        loss_s.append(dih_loss.detach()[None])
        
        return tot_loss, torch.cat(loss_s, dim=0)

    def calc_acc(self, prob, dist, idx_pdb):
        B = idx_pdb.shape[0]
        L = idx_pdb.shape[1] # (B, L)
        seqsep = torch.abs(idx_pdb[:,:,None] - idx_pdb[:,None,:]) + 1
        mask = seqsep > 24
        mask = torch.triu(mask.float())
        #
        cnt_ref = dist < 12
        cnt_ref = cnt_ref.float() * mask
        #
        cnt_pred = prob[:,:12,:,:].sum(dim=1) * mask
        #
        top_pred = torch.topk(cnt_pred.view(B,-1), L)
        kth = top_pred.values.min(dim=-1).values
        tmp_pred = list()
        for i_batch in range(B):
            tmp_pred.append(cnt_pred[i_batch] > kth[i_batch])
        cnt_pred = torch.stack(tmp_pred, dim=0)
        cnt_pred = cnt_pred.float()*mask
        #
        condition = torch.logical_and(cnt_pred==cnt_ref, cnt_ref==torch.ones_like(cnt_ref))
        n_good = condition.float().sum()
        n_total = (cnt_ref == torch.ones_like(cnt_ref)).float().sum() + 1e-9
        n_total_pred = (cnt_pred == torch.ones_like(cnt_pred)).float().sum() + 1e-9
        prec = n_good / n_total_pred
        recall = n_good / n_total
        F1 = 2.0*prec*recall / (prec+recall+1e-9)
        return torch.stack([prec, recall, F1])

    def load_model(self, model, optimizer, scheduler, scaler, model_name, rank, suffix='last', resume_train=False):
        chk_fn = "models/%s_%s.pt"%(model_name, suffix)
        loaded_epoch = -1
        best_valid_loss = 999999.9
        if not os.path.exists(chk_fn):
            return -1, best_valid_loss
        map_location = {"cuda:%d"%0: "cuda:%d"%rank}
        checkpoint = torch.load(chk_fn, map_location=map_location)
        rename_model = False
        for param in model.module.state_dict():
            if param not in checkpoint['model_state_dict']:
                rename_model=True
                break
        new_chk = checkpoint['model_state_dict']
        model.module.load_state_dict(new_chk, strict=False)
        if resume_train and (not rename_model):
            loaded_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                scheduler.last_epoch = loaded_epoch + 1
            #if 'best_loss' in checkpoint:
            #    best_valid_loss = checkpoint['best_loss']
        return loaded_epoch, best_valid_loss

    def checkpoint_fn(self, model_name, description):
        if not os.path.exists("models"):
            os.mkdir("models")
        name = "%s_%s.pt"%(model_name, description)
        return os.path.join("models", name)

    def run_model_training(self, world_size):
        #self.train_model(0, world_size)
        mp.spawn(self.train_model, args=(world_size,), nprocs=world_size, join=True)

    def train_model(self, rank, world_size):
        #print ("running ddp on rank %d, world_size %d"%(rank, world_size))
        torch.cuda.set_device("cuda:%d"%rank)
        DDP_setup(rank, world_size, self.port)

        #define dataset & data loader
        train_dict, valid_dict = get_train_valid_set(self.loader_param)
        self.n_train = len(train_dict.keys())
        self.n_valid = len(valid_dict.keys())
        
        train_set = Dataset(list(train_dict.keys()), loader_tbm, train_dict, self.loader_param)
        valid_set = Dataset(list(valid_dict.keys()), loader_tbm, valid_dict, self.loader_param)
        #
        train_sampler = data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
        valid_sampler = data.distributed.DistributedSampler(valid_set, num_replicas=world_size, rank=rank)
       
        train_loader = data.DataLoader(train_set, sampler=train_sampler, collate_fn=partial(tbm_collate_fn, params=self.loader_param), batch_size=self.batch_size, **LOAD_PARAM)
        valid_loader = data.DataLoader(valid_set, sampler=valid_sampler, collate_fn=partial(tbm_collate_fn, params=self.loader_param), **LOAD_PARAM)
        
        # define model
        model = TrunkModule(**self.model_param).to(rank)
        for n, p in model.named_parameters():
            #if "feat_extractor" in n: 
            #    if p.dim() > 1:
            #        nn.init.xavier_uniform_(p)
            if "encoder" in n: 
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            elif "str2str" in n: 
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            elif "str2msa" in n: 
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            #if "_emb" in n:
            #    p.requires_grad = False
            #elif "initial" in n:
            #    p.requires_grad = False
            #elif "iter_block_1" in n:
            #    tmp = n.split('.')[2]
            #    if int(tmp) == self.model_param['n_module']-1:
            #        continue
            #    p.requires_grad = False

        ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False)
        if rank == 0:
            print ("# of parameters:", count_parameters(ddp_model))
        
        # define optimizer and scheduler
        opt_params = add_weight_decay(ddp_model, self.l2_coeff)
        #optimizer = torch.optim.Adam(opt_params, lr=self.init_lr)
        optimizer = torch.optim.AdamW(opt_params, lr=self.init_lr)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.step_lr, gamma=0.5)
        scheduler = get_linear_schedule_with_warmup(optimizer, 16000, 200000)
        #scheduler = get_linear_schedule_with_warmup(optimizer, 0, 100000)
        #scheduler = get_linear_schedule_with_warmup(optimizer, 16000, 100000)
        scaler = torch.cuda.amp.GradScaler()
       
        # load model
        loaded_epoch, best_valid_loss = self.load_model(ddp_model, optimizer, scheduler, scaler, 
                                                        self.model_name, rank, resume_train=True)
        if loaded_epoch >= self.n_epoch:
            DDP_cleanup()
            return
        for epoch in range(loaded_epoch+1, self.n_epoch):
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
          
            train_tot, train_loss, train_acc = self.train_cycle(ddp_model, train_loader, optimizer, scheduler, scaler, rank, epoch)
            
            valid_tot, valid_loss, valid_acc = self.valid_cycle(ddp_model, valid_loader, rank, epoch)

            if rank == 0: # save model
                if valid_tot < best_valid_loss:
                    best_valid_loss = valid_tot
                    torch.save({'epoch': epoch,
                                #'model_state_dict': ddp_model.state_dict(),
                                'model_state_dict': ddp_model.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'scaler_state_dict': scaler.state_dict(),
                                'best_loss': best_valid_loss,
                                'train_loss': train_loss,
                                'train_acc': train_acc,
                                'valid_loss': valid_loss,
                                'valid_acc': valid_acc},
                                self.checkpoint_fn(self.model_name, 'best'))
            
            
                torch.save({'epoch': epoch,
                            #'model_state_dict': ddp_model.state_dict(),
                            'model_state_dict': ddp_model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'train_loss': train_loss,
                            'train_acc': train_acc,
                            'valid_loss': valid_loss,
                            'valid_acc': valid_acc,
                            'best_loss': best_valid_loss},
                            self.checkpoint_fn(self.model_name, 'last'))

        DDP_cleanup()

    def train_cycle(self, ddp_model, train_loader, optimizer, scheduler, scaler, rank, epoch):
        # Turn on training mode
        ddp_model.train()
        
        # clear gradients
        optimizer.zero_grad()

        start_time = time.time()
        
        # For intermediate logs
        local_tot = 0.0
        local_loss = None
        local_acc = None
        train_tot = 0.0
        train_loss = None
        train_acc = None

        counter = 0
        
        for msa, msa_full, true_crds, idx_pdb, xyz_t, t1d, t0d in train_loader:
            # transfer inputs to device
            B, N, L = msa.shape
            msa = msa.to(rank).long() # (B, N, L)
            mask = torch.rand(B, N, L, device=msa.device) < 0.15 # masking 15% of MSA inputs
            msa_masked = torch.where(mask, torch.full_like(msa, 21), msa)
            msa_full = msa_full.to(rank).long()
            idx_pdb = idx_pdb.to(rank).long() # (B, L)
            true_crds = true_crds.to(rank).float() # (B, L, 3, 3)
            c6d, _ = xyz_to_c6d(true_crds)
            c6d = c6d_to_bins2(c6d)
            tors = xyz_to_bbtor(true_crds)
            #
            # template features
            xyz_t = xyz_t.to(rank).float()
            t1d = t1d.to(rank).float()
            t0d = t0d.to(rank).float()
            t2d = xyz_to_t2d(xyz_t, t0d)
            #
            seq = msa[:,0] # first sequence = target sequence

            counter += 1
          
            if counter%self.ACCUM_STEP != 0:
                with ddp_model.no_sync():
                    with torch.cuda.amp.autocast(enabled=True):
                        logit_s, logit_aa_s, logit_tor_s, pred_crds, pred_lddts = ddp_model(msa_masked, msa_full, seq, idx_pdb,
                                                                   t1d=t1d, t2d=t2d,
                                                                   use_transf_checkpoint=True)
                        prob = self.active_fn(logit_s[0]) # distogram
                        acc_s = self.calc_acc(prob, c6d[...,0], idx_pdb)

                        loss, loss_s = self.calc_loss(logit_s, c6d,
                                logit_aa_s, msa, mask,
                                logit_tor_s, tors,    
                                pred_crds, true_crds, pred_lddts, idx_pdb, **self.loss_param)
                    loss = loss / self.ACCUM_STEP
                    scaler.scale(loss).backward()
            else:
                with torch.cuda.amp.autocast(enabled=True):
                    logit_s, logit_aa_s, logit_tor_s, pred_crds, pred_lddts = ddp_model(msa_masked, msa_full, seq, idx_pdb,
                                                               t1d=t1d, t2d=t2d,
                                                               use_transf_checkpoint=True)
                    prob = self.active_fn(logit_s[0]) # distogram
                    acc_s = self.calc_acc(prob, c6d[...,0], idx_pdb)

                    loss, loss_s = self.calc_loss(logit_s, c6d,
                            logit_aa_s, msa, mask,
                            logit_tor_s, tors,    
                            pred_crds, true_crds, pred_lddts, idx_pdb, **self.loss_param)
                loss = loss / self.ACCUM_STEP
                scaler.scale(loss).backward()
                # gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()
                skip_lr_sched = (scale != scaler.get_scale())
                optimizer.zero_grad()
                if not skip_lr_sched:
                    scheduler.step()
            
            ## check parameters with no grad
            #if rank == 0:
            #    for n, p in ddp_model.named_parameters():
            #        if p.grad is None and p.requires_grad is True:
            #            print('Parameter not used:', n, p.shape)  # prints unused parameters. Remove them from your model
            

            local_tot += loss.detach()*self.ACCUM_STEP
            if local_loss == None:
                local_loss = torch.zeros_like(loss_s.detach())
                local_acc = torch.zeros_like(acc_s.detach())
            local_loss += loss_s.detach()
            local_acc += acc_s.detach()
            
            train_tot += loss.detach()*self.ACCUM_STEP
            if train_loss == None:
                train_loss = torch.zeros_like(loss_s.detach())
                train_acc = torch.zeros_like(acc_s.detach())
            train_loss += loss_s.detach()
            train_acc += acc_s.detach()

            
            if counter % N_PRINT_TRAIN == 0:
                if rank == 0:
                    max_mem = torch.cuda.max_memory_allocated()/1e9
                    train_time = time.time() - start_time
                    local_tot /= float(N_PRINT_TRAIN)
                    local_loss /= float(N_PRINT_TRAIN)
                    local_acc /= float(N_PRINT_TRAIN)
                    
                    local_tot = local_tot.cpu().detach()
                    local_loss = local_loss.cpu().detach().numpy()
                    local_acc = local_acc.cpu().detach().numpy()

                    sys.stdout.write("Local: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f | Max mem %.4f\n"%(\
                            epoch, self.n_epoch, counter*self.batch_size*N_GPU, self.n_train, train_time, local_tot, \
                            " ".join(["%8.4f"%l for l in local_loss]),\
                            local_acc[0], local_acc[1], local_acc[2], max_mem))
                    sys.stdout.flush()
                    local_tot = 0.0
                    local_loss = None 
                    local_acc = None 
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        # write total train loss
        train_tot /= float(counter * N_GPU)
        train_loss /= float(counter * N_GPU)
        train_acc  /= float(counter * N_GPU)

        dist.all_reduce(train_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_acc, op=dist.ReduceOp.SUM)
        train_tot = train_tot.cpu().detach()
        train_loss = train_loss.cpu().detach().numpy()
        train_acc = train_acc.cpu().detach().numpy()
        if rank == 0:
            
            train_time = time.time() - start_time
            sys.stdout.write("Train: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f\n"%(\
                    epoch, self.n_epoch, self.n_train, self.n_train, train_time, train_tot, \
                    " ".join(["%8.4f"%l for l in train_loss]),\
                    train_acc[0], train_acc[1], train_acc[2]))
            sys.stdout.flush()
            
        return train_tot, train_loss, train_acc

    def valid_cycle(self, ddp_model, valid_loader, rank, epoch):
        valid_tot = 0.0
        valid_loss = None
        valid_acc = None
        
        start_time = time.time()
        
        with torch.no_grad(): # no need to calculate gradient
            ddp_model.eval() # change it to eval mode
            for msa, msa_full, true_crds, idx_pdb, xyz_t, t1d, t0d in valid_loader:
                # transfer inputs to device
                B, N, L = msa.shape
                msa = msa.to(rank).long() # (B, N, L)
                mask = torch.rand(B, N, L, device=msa.device) < 0.15 # masking 15% of MSA inputs
                msa_masked = torch.where(mask, torch.full_like(msa, 21), msa)
                msa_full = msa_full.to(rank).long()
                idx_pdb = idx_pdb.to(rank).long() # (B, L)
                true_crds = true_crds.to(rank).float() # (B, L, 3, 3)
                c6d, _ = xyz_to_c6d(true_crds)
                c6d = c6d_to_bins2(c6d)
                tors = xyz_to_bbtor(true_crds)
                #
                # template features
                xyz_t = xyz_t.to(rank).float()
                t1d = t1d.to(rank).float()
                t0d = t0d.to(rank).float()
                t2d = xyz_to_t2d(xyz_t, t0d)
                #
                seq = msa[:,0] # first sequence = target sequence
              
                logit_s, logit_aa_s, logit_tor_s, pred_crds, pred_lddts = ddp_model(msa_masked, msa_full, seq, idx_pdb,
                                                           t1d=t1d, t2d=t2d,
                                                           use_transf_checkpoint=True)
                prob = self.active_fn(logit_s[0]) # distogram
                acc_s = self.calc_acc(prob, c6d[...,0], idx_pdb)

                loss, loss_s = self.calc_loss(logit_s, c6d,
                        logit_aa_s, msa, mask,
                        logit_tor_s, tors,    
                        pred_crds, true_crds, pred_lddts, idx_pdb, **self.loss_param)
                
                valid_tot += loss.detach()
                if valid_loss == None:
                    valid_loss = torch.zeros_like(loss_s.detach())
                    valid_acc = torch.zeros_like(acc_s.detach())
                valid_loss += loss_s.detach()
                valid_acc += acc_s.detach()

            
        valid_tot /= float(self.n_valid)
        valid_loss /= float(self.n_valid)
        valid_acc /= float(self.n_valid)
        
        dist.all_reduce(valid_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_acc, op=dist.ReduceOp.SUM)
       
        valid_tot = valid_tot.cpu().detach().numpy()
        valid_loss = valid_loss.cpu().detach().numpy()
        valid_acc = valid_acc.cpu().detach().numpy()
        
        if rank == 0:
            
            train_time = time.time() - start_time
            sys.stdout.write("Valid: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f\n"%(\
                    epoch, self.n_epoch, self.n_valid, self.n_valid, train_time, valid_tot, \
                    " ".join(["%8.4f"%l for l in valid_loss]),\
                    valid_acc[0], valid_acc[1], valid_acc[2])) 
            sys.stdout.flush()
        return valid_tot, valid_loss, valid_acc
    

if __name__ == "__main__":
    from arguments import get_args
    args, model_param, loader_param, loss_param = get_args()

    mp.freeze_support()
    train = Trainer(model_name=args.model_name,
                    n_epoch=args.num_epochs, step_lr=args.step_lr, lr=args.lr, l2_coeff=1.0e-2,
                    port=args.port, model_param=model_param, loader_param=loader_param, 
                    loss_param=loss_param, 
                    batch_size=args.batch_size,
                    accum_step=args.accum)
    train.run_model_training(torch.cuda.device_count())
