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

def encode_confusion_3(X,Y,Z):
    v0=(X.diag()*Y.diag()*Z.diag()).sum()
    
    v1=(X.diag()*Y.diag()*Z.sum(-1)).sum()
    v2=(X.diag()*Y.diag()*Z.sum(-2)).sum()
    
    v3=(X.diag()*(Y*Z).sum(dim=-1)).sum()
    v4=(X.diag()*(Y*Z).sum(dim=-2)).sum()
    v5=(X.diag()*(Y*Z.t()).sum(dim=-1)).sum()
    
    v6=(X.diag()*torch.mm(Y,Z.diag().view(-1,1))).sum()
    v7=(X.diag()*torch.mm(Y.t(),Z.diag().view(-1,1))).sum()
    
    v8=(X*Y*Z).sum()
    v9=(X*Y*Z.t()).sum()
    
    v10=(X.diag()*Y.sum(-1)*Z.sum(-1)).sum()
    v11=(X.diag()*Y.sum(-2)*Z.sum(-1)).sum()
    v12=(X.diag()*Y.sum(-2)*Z.sum(-2)).sum()
    
    
    v13=(X.diag()*torch.mm(Y,Z.sum(dim=-1,keepdim=True))).sum()
    v14=(X.diag()*torch.mm(Y,Z.sum(dim=-2).view(-1,1))).sum()
    
    v15=torch.mm(X.t()*Y.t(),Z.sum(dim=-1,keepdim=True)).sum()
    v16=torch.mm(X*Y.t(),Z.sum(dim=-1,keepdim=True)).sum()
    v17=torch.mm(X*Y,Z.sum(dim=-1,keepdim=True)).sum()
    
    v18=torch.mm(X.t()*Y.t(),Z.sum(dim=-2).view(-1,1)).sum()
    v19=torch.mm(X*Y.t(),Z.sum(dim=-2).view(-1,1)).sum()
    v20=torch.mm(X*Y,Z.sum(dim=-2).view(-1,1)).sum()
    
    v21=(X*torch.mm(Y,Z.t())).sum()
    h=torch.stack((v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21),dim=0)
    return h
    
    
    

def characterize(interface,data=None,params=None):
    print('Starting trigger characterization')
    nclasses=interface.nclasses()
    fvs=[];
    for i,data_t in enumerate(data):
        for j,(ex_clean,ex_triggered) in enumerate(data_t):
            #print('Extracting features for trigger %d/%d, %d/%d         '%(i,len(data),j,len(data_t)),end='\r')
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
            
            '''
            for X in [fv0,fv1,fv2]:
                for Y in [fv0,fv1,fv2]:
                    for Z in [fv0,fv1,fv2]:
                        h.append(encode_confusion_3(X,Y,Z))
            '''
            
            h=torch.cat(h,dim=0); # 108-dim
            fvs.append(h)
    
    fvs=torch.stack(fvs,dim=0)
    fvs=fvs.view(len(data),-1,108) #+27*22
    
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
    
    '''
    if params.preextracted:
        dataset=[torch.load(os.path.join('data_r13_trinity_v1',fname)) for fname in os.listdir(params.data) if fname.endswith('.pt')];
        dataset=db.Table.from_rows(dataset)
        return dataset
    '''
    
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
    print('Starting trigger search')
    #Load as many clean examples as possible
    data=interface.load_examples()
    data=data+interface.more_clean_examples()
    #A list of random-colored patches as the trigger
    #triggers=[x['trigger'] for x in interface.get_triggers()]
    #if len(triggers)==0:
    #    trigger=torch.Tensor([1,0,1,1]).view(4,1,1)
    #    triggers.append(trigger)
    
    
    triggers=[]
    for r in [0,0.5,1.0]:
        for g in [0,0.5,1.0]:
            for b in [0,0.5,1.0]:
                trigger=torch.Tensor([r,g,b,1]).view(4,1,1)
                triggers.append(trigger)
    
    #Apply each trigger on clean examples at multiple random locations, as the output of trigger search
    pairs=[]
    n=0
    for trigger in triggers:
        pairs_t=[]
        for i,ex in enumerate(data):
            #print('processing example %d/%d  '%(i,len(data)),end='\r')
            for j,box in enumerate(ex['gt']):
                ex2=interface.trojan_examples([ex],trigger,boxid=j)[0]
                pairs_t.append((ex,ex2))
                n+=1
            
            ex2=interface.trojan_examples([ex],trigger)[0]
            pairs_t.append((ex,ex2))
            n+=1
        
        pairs.append(pairs_t)
    
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
    default_params.out='data_r13_trinity_pred2c'
    params=smartparse.parse(default_params);
    params.argv=sys.argv;
    
    extract_dataset(os.path.join(helper.root(),'models'),ts_engine,params);
    
