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

def encode_confusion(X,Y):
    v0=X.sum()
    v1=X.diag().sum()
    v2=Y.sum()
    v3=Y.diag().sum()
    
    v4=(X.diag()*Y.diag()).sum()
    v5=(X.diag()*Y.sum(dim=-1)).sum()
    v6=(X.diag()*Y.sum(dim=-2)).sum()
    v7=(X*Y).sum()
    v8=(X*Y.t()).sum()
    
    v9=(X.sum(dim=-1)*Y.sum(dim=-1)).sum()
    v10=(X.sum(dim=-2)*Y.sum(dim=-1)).sum()
    v11=(X.sum(dim=-2)*Y.sum(dim=-2)).sum()
    
    h=torch.stack((v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11),dim=0)
    return h

def characterize(interface,data=None,params=None):
    nclasses=interface.nclasses()
    fvs=[];
    for j,(ex_clean,ex_triggered) in enumerate(data):
        print('Extracting features for trigger %d/%d         '%(j,len(data)),end='\r')
        #print(ex_clean.keys(),ex_triggered.keys())
        pred_clean=interface.eval(ex_clean)
        pred_triggered=interface.eval(ex_triggered)
        
        gt=helper.prepare_boxes_as_prediction(ex_clean['gt'])
        fv0=helper.compare_boxes(gt,pred_clean,nclasses)
        fv1=helper.compare_boxes(gt,pred_triggered,nclasses)
        fv2=helper.compare_boxes(pred_clean,pred_triggered,nclasses)
        
        h=[]
        for X in [fv0,fv1,fv2]:
            for Y in [fv0,fv1,fv2]:
                h.append(encode_confusion(X,Y))
        
        h=torch.cat(h,dim=0); # 108-dim
        fvs.append(h)
    
    fvs=torch.stack(fvs,dim=0)
    fvs=fvs.view(len(data),108)
    
    print(fvs.shape)
    return {'fvs':fvs};

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
    #Load as many clean examples as possible
    data=interface.load_examples()
    data=data+interface.more_clean_examples()
    
    pairs=[]
    n=0
    import diversity_exp as trigger_search
    for i in range(len(data)):
        im=data[i]['image']
        triggers2=trigger_search.run(interface,data[i]);
        for trigger in triggers2:
            import arch.meta_poly7 as arch
            im2=arch.apply(trigger,im.cuda())
            d=copy.deepcopy(data[i]);
            d['image']=im2.data.cpu();
            
            pairs.append((data[i],d))
            n+=1
    
    print('total %d triggered pairs  '%(n))
    return pairs

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
    default_params.out='data_r13_trinity_pred3'
    params=smartparse.parse(default_params);
    params.argv=sys.argv;
    
    extract_dataset(os.path.join(helper.root(),'models'),ts_engine,params);
    
