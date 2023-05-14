import os
import sys
import time
from pathlib import Path
import importlib
import math
import copy

import torch
import util.db as db
import util.smartparse as smartparse
import helper_r13_v2 as helper


#Interface spec
# Interface.model: nn.Module
#   support zero_grad()
#   support named_parameters()

# load_examples: returns a table of examples
# eval: return correctness: loss value of one or multiple examples

'''
def compute_grad(interface,example):
    interface.model.zero_grad()
    s=interface.eval(example)
    loss=s
    loss.backward()
    params=interface.model.named_parameters();
    g=[];
    for k,p in params:
        #if not p.grad is None:
        g.append(p.grad.data.clone().cpu())
        #print(p.grad.max(),p.grad.min())
        #else:
        #    g.append(None)
    
    return g
'''

def compute_grad(interface,example):
    interface.model.zero_grad()
    loss=interface.eval_loss(example)
    loss.backward()
    g=[]
    for name,param in interface.model.named_parameters():
        if param.grad is None:
            g.append(param.data.cpu()*0)
        else:
            g.append(param.grad.data.cpu()) 
    
    return g


def analyze_tensor(w,params=None):
    default_params=smartparse.obj();
    default_params.bins=100;
    params=smartparse.merge(params,default_params)
    
    if w is None:
        return torch.Tensor(params.bins*2).fill_(0)
    else:
        q=torch.arange(params.bins).float().cuda()/(params.bins-1)
        hw=torch.quantile(w.view(-1).float(),q.to(w.device)).contiguous().cpu();
        hw_abs=torch.quantile(w.abs().view(-1).float(),q.to(w.device)).contiguous().cpu();
        fv=torch.cat((hw,hw_abs),dim=0);
        return fv;

def grad2fv(g,params=None):
    fvs=[analyze_tensor(w.cuda(),params) for w in g]
    fvs=torch.stack(fvs,dim=0);
    return fvs;

def characterize(interface,data=None,params=None):
    if data is None:
        data=interface.load_examples()
    
    #Compute grads for examples
    fvs=[compute_grad(interface,data[i]) for i in range(len(data))]
    #group by layer
    fvs=[torch.stack([fv[i].view(-1) for fv in fvs],dim=0) for i in range(len(fvs[0]))]
    
    
    with torch.no_grad():
        hx=[];
        hx2=[];
        hxy=[];
        for fv in fvs:
            hx.append(fv.cuda().mean(dim=-1)) 
            hx2.append((fv.cuda()**2).mean(dim=-1)) 
            hxy.append(torch.mm(fv.cuda(),fv.cuda().t())/fv.shape[-1])
        
        hx=torch.stack(hx,dim=0)
        hx2=torch.stack(hx2,dim=0)
        hxy=torch.stack(hxy,dim=0)
    
    return {'hx':hx,'hx2':hx2,'hxy':hxy}


def extract_fv(interface,ts_engine,params=None):
    data=ts_engine(interface,params=params);
    fvs=characterize(interface,data,params);
    return fvs


#Extract dataset from a folder of training models
def extract_dataset(models_dirpath,ts_engine,params=None):
    default_params=smartparse.obj();
    default_params.rank=0
    default_params.world_size=1
    default_params.out=''
    default_params.preextracted=False
    
    params=smartparse.merge(params,default_params)
    
    if params.preextracted:
        dataset=[torch.load(os.path.join('data_r13_trinity_v1',fname)) for fname in os.listdir(params.data) if fname.endswith('.pt')];
        dataset=db.Table.from_rows(dataset)
        return dataset
    
    t0=time.time()
    models=os.listdir(models_dirpath);
    models=sorted(models)
    models=[(i,x) for i,x in enumerate(models)]
    
    dataset=[];
    for i,fname in models[params.rank::params.world_size]:
        folder=os.path.join(models_dirpath,fname);
        interface=helper.engine(folder=folder,params=params)
        fvs=extract_fv(interface,ts_engine,params=params);
        
        #Load GT
        fname_gt=os.path.join(folder,'ground_truth.csv');
        f=open(fname_gt,'r');
        for line in f:
            line.rstrip('\n').rstrip('\r')
            label=int(line);
            break;
        
        f.close();
        
        data={'params':vars(params),'model_name':fname,'model_id':i,'label':label};
        fvs.update(data);
        
        if not params.out=='':
            Path(params.out).mkdir(parents=True, exist_ok=True);
            torch.save(fvs,'%s/%d.pt'%(params.out,i));
        
        dataset.append(fvs);
        print('Model %d(%s), time %f'%(i,fname,time.time()-t0));
    
    dataset=db.Table.from_rows(dataset);
    return dataset;

def generate_probe_set(models_dirpath,params=None):
    models=os.listdir(models_dirpath)
    models=sorted(models)
    
    all_data=[];
    for m in models[:len(models)//2]:
        interface=helper.engine(os.path.join(models_dirpath,m))
        data=interface.load_examples()
        data=list(data.rows())
        
        try:
            data2=interface.load_examples(os.path.join(models_dirpath,m,'poisoned-example-data'))
            data2=list(data2.rows())
            data+=data2
        except:
            pass;
        
        all_data+=data;
    
    all_data=db.Table.from_rows(all_data)
    return all_data



def ts_engine(interface,additional_data='enum.pt',params=None):
    import diversity_exp2 as trigger_search
    data=list(interface.load_examples().rows())
    data=data+interface.more_clean_examples()
    
    data2=interface.load_poisoned_examples()
    if not data2 is None:
        data2=interface.target_to_source(data2)
        data=data+list(data2.rows())[0:1]
    
    '''
    new_data=[];
    for i in range(len(data)):
        im=data[i]['image']
        
        triggers2=trigger_search.run_color(interface,data[i]);
        for trigger in triggers2:
            import arch.meta_color7 as arch
            im2=arch.apply(trigger,im.cuda())
            d=copy.deepcopy(data[i]);
            d['image']=im2.data.cpu();
            #d['trigger']=trigger.data.cpu()
            new_data.append(d);
    
    data=db.Table.from_rows(list(data.rows())+new_data)
    '''
    data=db.Table.from_rows(data)
    
    return data



def predict(ensemble,fvs):
    scores=[];
    with torch.no_grad():
        for i in range(len(ensemble)):
            params=ensemble[i]['params'];
            arch=importlib.import_module(params.arch);
            net=arch.new(params);
            net.load_state_dict(ensemble[i]['net'],strict=True);
            net=net.cuda();
            net.eval();
            
            s=net.logp(fvs).data.cpu();
            s=s*math.exp(-ensemble[i]['T']);
            scores.append(s)
    
    s=sum(scores)/len(scores);
    s=torch.sigmoid(s); #score -> probability
    trojan_probability=float(s);
    return trojan_probability

if __name__ == "__main__":
    import os
    default_params=smartparse.obj();
    default_params.out='data_r13_trinity_cheat'
    params=smartparse.parse(default_params);
    params.argv=sys.argv;
    
    extract_dataset(os.path.join(helper.root(),'models'),ts_engine,params);
    
