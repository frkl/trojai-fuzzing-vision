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
import torch.nn.functional as F

def analyze_tensor(x,y):
    x=x.cuda().contiguous()
    y=y.cuda().contiguous()
    h=[];
    if len(x.shape)>=1:
        h.append(x.mean())
        h.append(x.std())
        h.append(y.mean())
        h.append(y.std())
    if len(x.shape)>=2:
        h.append(x.mean(dim=0).std())
        h.append(x.mean(dim=1).std())
        h.append(y.mean(dim=0).std())
        h.append(y.mean(dim=1).std())
    if  len(x.shape)>=3:
        h.append(x.mean(dim=2).std())
        h.append(x.mean(dim=(0,1)).std())
        h.append(x.mean(dim=(0,2)).std())
        h.append(x.mean(dim=(1,2)).std())
        h.append(y.mean(dim=2).std())
        h.append(y.mean(dim=(0,1)).std())
        h.append(y.mean(dim=(0,2)).std())
        h.append(y.mean(dim=(1,2)).std())
    
    h=h+[h[0]*0]*(16-len(h))
    if len(x.shape)>=1:
        if x.shape[0]==y.shape[0] and x.shape[1]==y.shape[1]:
            h.append((x*y).mean());
        else:
            h.append(h[0]*0)
        
    elif len(x.shape)>=2:
        if x.shape[0]==y.shape[0] and x.shape[1]==y.shape[1]:
            h.append((x*y).mean());
        else:
            h.append(h[0]*0)
        
        if x.shape[0]==y.shape[0]:
            h.append((x.mean(dim=1)*y.mean(dim=1)).mean())
        else:
            h.append(h[0]*0)
        
        if x.shape[1]==y.shape[1]:
            h.append((x.mean(dim=0)*y.mean(dim=0)).mean())
        else:
            h.append(h[0]*0)
    
    elif len(x.shape)>=3:
        if x.shape[0]==y.shape[0] and x.shape[1]==y.shape[1] and x.shape[1]==y.shape[1]:
            h.append((x*y).mean());
        else:
            h.append(h[0]*0)
        
        
        if x.shape[0]==y.shape[0]:
            h.append((x.mean(dim=(1,2))*y.mean(dim=(1,2))).mean())
        else:
            h.append(h[0]*0)
        
        if x.shape[1]==y.shape[1]:
            h.append((x.mean(dim=(0,2))*y.mean(dim=(0,2))).mean())
        else:
            h.append(h[0]*0)
        
        if x.shape[2]==y.shape[2]:
            h.append((x.mean(dim=(0,1))*y.mean(dim=(0,1))).mean())
        else:
            h.append(h[0]*0)
    
        if x.shape[0]==y.shape[0] and x.shape[1]==y.shape[1]:
            h.append((x.mean(dim=2)*y.mean(dim=2)).mean())
        else:
            h.append(h[0]*0)
        
        if x.shape[2]==y.shape[2] and x.shape[0]==y.shape[0]:
            h.append((x.mean(dim=1)*y.mean(dim=1)).mean())
        else:
            h.append(h[0]*0)
        
        if x.shape[2]==y.shape[2] and x.shape[1]==y.shape[1]:
            h.append((x.mean(dim=0)*y.mean(dim=0)).mean())
        else:
            h.append(h[0]*0)
    
    h=h+[h[0]*0]*(23-len(h))
        
    
    
    h=torch.stack(h,dim=0).view(-1);
    return h;

def analyze_tensor(x,y):
    x=x.cuda().contiguous()
    y=y.cuda().contiguous()
    h=[];
    x=F.normalize(x.view(-1),dim=-1)
    y=F.normalize(y.view(-1),dim=-1)
    if not len(x)==len(y):
        h=x.sum().cpu()*0+1
    else:
        h=(x*y).sum().cpu()
    
    return h;

def characterize(interface,data1=None,data2=None,params=None):
    fvs=[];
    for i in range(len(data1)):
        h1=interface.eval_hidden(data1[i])
        h2=interface.eval_hidden(data2[i])
        
        v=[analyze_tensor(h1[j],h2[j]) for j in range(len(h1))]
        v=torch.stack(v,dim=0)
        fv=v
        
        fvs.append(fv)
    
    fvs=torch.stack(fvs,dim=0)
    print(fvs.shape)
    return {'fvs':fvs}

def extract_fv(interface,ts_engine,params=None):
    data1,data2=ts_engine(interface,params=params);
    fvs=characterize(interface,data1,data2,params);
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
    data1=interface.load_examples()
    data=interface.load_poisoned_examples()
    if len(data)>0:
        data=interface.target_to_source(data)
    
    data1+=data
    data2=interface.replace(data1)
    if data2 is None:
        data2=data1
    
    return data1,data2



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
    default_params.out='data_r13_trinity_cheat_embed'
    params=smartparse.parse(default_params);
    params.argv=sys.argv;
    
    extract_dataset(os.path.join(helper.root(),'models'),ts_engine,params);
    
