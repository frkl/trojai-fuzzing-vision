
import importlib
import math
import time
import sys
import json



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


from hyperopt import hp, tpe, fmin
import sklearn.metrics



import util.db as db
import util.smartparse as smartparse
import util.session_manager as session_manager



class HP_config:
    def __init__(self):
        self.data=[];
    
    def add(self,method,name,options=None,low=None,high=None,q=1):
        #dtype
        if method=='qloguniform' or method=='quniform':
            dtype=int
            spec=(low,high)
        elif method=='choice':
            dtype=str
            spec=options
        else:
            dtype=float
            spec=(low,high)
        
        #instantiate
        if method=='qloguniform':
            x=getattr(hp, method)(name,low=math.log(low),high=math.log(high),q=q);
        elif method=='loguniform':
            x=getattr(hp, method)(name,low=math.log(low),high=math.log(high));
        elif method=='quniform':
            x=getattr(hp, method)(name,low=low,high=high,q=q);
        elif method=='uniform':
            x=getattr(hp, method)(name,low=low,high=high);
        elif method=='choice':
            x=getattr(hp, method)(name,options);
        
        self.data.append((name,x,dtype,spec));
    
    def params(self):
        return [x[1] for x in self.data];
    
    def parse(self,*args):
        return {x[0]:x[2](args[i]) for i,x in enumerate(self.data)}
    
    def export(self):
        return [(x[0],x[2],x[3]) for x in self.data];



class Dataset:
    def __init__(self,split):
        self.data=split;
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,i):
        data=self.data[i];
        return data

def collate(x):
    x=db.Table.from_rows(x)
    return x;

def metrics(scores,gt):
    auc=float(sklearn.metrics.roc_auc_score(gt.long().numpy(),scores.numpy()));
    sgt=F.logsigmoid(scores*(gt*2-1))
    ce=-sgt.mean()
    cestd=sgt.std()/len(sgt)**0.5;
    return auc,float(ce),float(cestd)

def train(crossval_splits,params):
    default_params=smartparse.obj();
    default_params.rank=0
    
    params=smartparse.merge(params,default_params)
    
    #Training
    arch=importlib.import_module(params.arch);
    t0=time.time();
    nets=[];
    for split_id,split in enumerate(crossval_splits):
        data_train,data_val,data_test=split;
        loader_train = DataLoader(Dataset(data_train.cuda()),collate_fn=collate,batch_size=params.batch,shuffle=True,drop_last=True,num_workers=0);
        net=arch.new(params).cuda();
        opt=optim.AdamW(net.parameters(),lr=params.lr,weight_decay=params.decay);
        
        for iter in range(params.epochs):
            net.train();
            for data_batch in loader_train:
                opt.zero_grad();
                data_batch.cuda();
                C=torch.LongTensor(data_batch['label']).cuda();
                
                #Cross entropy
                scores_i=net.logp(data_batch)
                loss=F.binary_cross_entropy_with_logits(scores_i,C.float());
                
                #Mutual information
                '''
                scores_i=net(data_batch);
                spos=scores_i.gather(1,C.view(-1,1)).mean();
                sneg=torch.exp(scores_i).mean();
                loss=-(spos-sneg+1);
                '''
                
                #L1-L2 
                #l2=torch.stack([(p**2).sum() for p in net.parameters()],dim=0).sum()
                #loss=loss+l2*params.decay;
                
                loss.backward();
                opt.step();
        
        net.eval();
        nets.append(net);
    
    #Calibration
    scores=[];
    gt=[];
    for split_id,split in enumerate(crossval_splits):
        data_train,data_val,data_test=split;
        net=nets[split_id];
        net.eval();
        loader_val = DataLoader(Dataset(data_val.cuda()),collate_fn=collate,batch_size=params.batch,num_workers=0);
        with torch.no_grad():
            for data_batch in loader_val:
                data_batch.cuda();
                scores_i=net.logp(data_batch);
                scores.append(scores_i.data.cpu());
                gt.append(torch.LongTensor(data_batch['label']));
    
    scores=torch.cat(scores,dim=0);
    gt=torch.cat(gt,dim=0);
    
    T=torch.Tensor(1).fill_(0).cuda();
    T.requires_grad_();
    opt2=optim.Adamax([T],lr=3e-2);
    for iter in range(500):
        opt2.zero_grad();
        loss=F.binary_cross_entropy_with_logits(scores.cuda()*torch.exp(-T),gt.float().cuda());
        loss.backward();
        opt2.step();
    
    T=float(T.data)
    
    #Eval
    scores=[];
    gt=[];
    model_name=[]
    for split_id,split in enumerate(crossval_splits):
        data_train,data_val,data_test=split;
        net=nets[split_id];
        net.eval();
        loader_test = DataLoader(Dataset(data_test.cuda()),collate_fn=collate,batch_size=params.batch,num_workers=0);
        with torch.no_grad():
            for data_batch in loader_test:
                data_batch.cuda();
                scores_i=net.logp(data_batch);
                scores.append(scores_i.data.cpu());
                gt.append(torch.LongTensor(data_batch['label']));
                model_name=model_name+data_batch['model_name'];
    
    scores=torch.cat(scores,dim=0);
    scores_T=scores*math.exp(-T)
    gt=torch.cat(gt,dim=0);
    
    auc,ce,cestd=metrics(scores,gt)
    _,ce_T,cestd_T=metrics(scores_T,gt)
    
    FN=[];
    FP=[];
    for i in range(len(gt)):
        if int(gt[i])==1 and float(scores[i])<=0:
            FN.append(model_name[i]);
        elif int(gt[i])==0 and float(scores[i])>=0:
            FP.append(model_name[i]);
    
    FN=sorted(FN);
    FP=sorted(FP);
    
    #Generate checkpoint
    ensemble=[{'net':net.state_dict(),'T':T,'params':params} for net in nets]
    
    print('AUC: %f, CE: %f + %f, CEpre: %f + %f, time %.2f'%(auc,ce_T,cestd_T,ce,cestd,time.time()-t0));
    return ensemble,(auc,ce_T,cestd_T,ce,cestd,FN,FP)
    

def split(dataset,params=None):
    default_params=smartparse.obj();
    default_params.splits=4;
    params=smartparse.merge(params,default_params)
    
    ind=torch.randperm(len(dataset)).tolist();
    crossval_splits=[];
    for split_id in range(params.splits):
        ind_test=ind[split_id::params.splits]
        ind_train=list(set(ind).difference(set(ind_test)))
        
        data_train=dataset.select_by_index(ind_train);
        data_test=dataset.select_by_index(ind_test);
        crossval_splits.append((data_train,data_test,data_test))
    
    return crossval_splits
    

best_so_far=1e10
def crossval_hyper(dataset,params):
    default_params=smartparse.obj();
    default_params.arch='arch.mlp_set5f';
    default_params.session_dir=None;
    params=smartparse.merge(params,default_params)
    
    session=session_manager.create_session(params);
    
    #Produce crossval splits
    crossval_splits=split(dataset,params);
    
    #Hyperparam search config
    hp_config=HP_config();
    hp_config.add('choice','arch',[params.arch]);
    hp_config.add('qloguniform','nh',low=16,high=512);
    hp_config.add('qloguniform','nh2',low=16,high=512);
    hp_config.add('qloguniform','nh3',low=16,high=512);
    hp_config.add('quniform','nlayers',low=1,high=12);
    hp_config.add('quniform','nlayers2',low=1,high=12);
    hp_config.add('quniform','nlayers3',low=1,high=12);
    hp_config.add('loguniform','margin',low=2,high=1e1);
    
    hp_config.add('qloguniform','epochs',low=3,high=300);
    hp_config.add('loguniform','lr',low=1e-5,high=1e-2);
    hp_config.add('loguniform','decay',low=1e-5,high=1e1);
    hp_config.add('qloguniform','batch',low=8,high=64);
    
    #HP search function
    t0=time.time();
    def run_crossval(p):
        global best_so_far
        params=hp_config.parse(*p);
        print('HP: %s'%(json.dumps(params)));
        params=smartparse.dict2obj(params);
        ensemble,perf=train(crossval_splits,params);
        #wrap hp_config into ensemble
        [x.update({'config':hp_config.export()}) for x in ensemble];
        
        auc,ce_T,cestd_T,ce,cestd,FN,FP=perf
        session.log('AUC: %f, CE: %f + %f, CEpre: %f + %f, time %.2f'%(auc,ce_T,cestd_T,ce,cestd,time.time()-t0));
        #session.log('FN: '+','.join(['%s'%i for i in FN]));
        #session.log('FP: '+','.join(['%s'%i for i in FP]));
        
        if ce_T<best_so_far:
            best_so_far=ce_T
            torch.save(ensemble,session.file('model.pt'))
        
        return ce_T
    
    #Launch HP search
    best=fmin(run_crossval,hp_config.params(),algo=tpe.suggest,max_evals=100000);


if __name__ == "__main__":
    import util.perm_inv as inv
    comps=[];
    for i in range(1,6):
        comps+=inv.generate_comps(i)
    
    inv_net1=inv.perm_inv2(comps,w=300,h=200)
    
    class vector_log(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.sign()*torch.log1p(x.abs())/10
        
        @staticmethod
        def backward(ctx, grad_output):
            x=ctx.saved_tensors
            return grad_output/(1+x.abs())/10
    
    vector_log = vector_log.apply

    import os
    default_params=smartparse.obj();
    default_params.data='data_r13_trinity_v0'
    params=smartparse.parse(default_params);
    params.argv=sys.argv;
    
    if params.data.endswith('.pt'):
        dataset=db.DB.load(params.data)
        dataset=dataset['table_ann']
    else:
        dataset=[torch.load(os.path.join(params.data,fname)) for fname in os.listdir(params.data) if fname.endswith('.pt')];
        dataset=db.Table.from_rows(dataset)
    
    '''
    with torch.no_grad():
        for i in range(len(dataset['fvs'])):
            fv=dataset['fvs'][i].cuda()
            #fv=vector_log(fv*1e3)
            fv=inv_net1(fv.permute(1,0,2))
            dataset['fvs'][i]=fv.cpu()
    '''
    
    crossval_hyper(dataset,params)
    







