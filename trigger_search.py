


import os
import sys
import torch
import time
import json
import jsonschema
import jsonpickle

import math

import torch.nn.functional as F
import util.db as db
import util.smartparse as smartparse

import helper_r10 as helper
import engine_objdet as engine

#Fuzzing call for TrojAI R9
def extract_fv(id=None, model_filepath=None, scratch_dirpath=None, examples_dirpath=None, params=None):
    t0=time.time();
    default_params=smartparse.obj();
    default_params.nbins=100
    default_params.szcap=4096
    params = smartparse.merge(params,default_params);
    
    if not id is None:
        model_filepath, scratch_dirpath, examples_dirpath=helper.get_paths(id);
    
    interface=engine.new(model_filepath);
    try:
        examples=interface.load_examples('./sample_images_1k',bsz=1000)
    except:
        examples=interface.load_examples('/sample_images_1k',bsz=1000)
    
    results=interface.inference(examples);
    for i,r in enumerate(results):
        r['annotation']=examples['annotation'][i]
    
    fvs=results;
    print('Inference done, time %.2f'%(time.time()-t0))
    return fvs

def pool(fvs):
    #Return a GTboxes by 100 matrix
    fvs_pool=[];
    for imid,data in enumerate(fvs):
        pred=data['pred']
        ann=data['annotation']
        for boxid,box in enumerate(ann):
            v=torch.zeros(100)+1e-7;
            v2=torch.zeros(100)+1e-7;
            v2[box['category_id']]=1;
            for boxid2,box2 in enumerate(pred['boxes'].tolist()):
                s=iou(box['bbox'],box2)
                if s>0:
                    score=float(pred['scores'][boxid2]);
                    label=int(pred['labels'][boxid2]);
                    v[label]=max(v[label],score*s)
            
            v=F.normalize(v,dim=0,p=1);
            fvs_pool.append(torch.cat((v,v2),dim=0));
    
    
    fvs_pool=torch.stack(fvs_pool,dim=0);
    return fvs_pool;

def pool2(fvs):
    #Return a GTboxes by 100 matrix
    fvs_pool=[];
    for imid,data in enumerate(fvs):
        pred=data['pred']
        ann=data['annotation']
        for boxid,box in enumerate(ann):
            for boxid2,box2 in enumerate(pred['boxes'].tolist()):
                s=iou(box['bbox'],box2)
                if s>0:
                    score=float(pred['scores'][boxid2]);
                    label=int(pred['labels'][boxid2]);
                    v[label]=max(v[label],score*s)
            
            v=F.normalize(v,dim=0,p=1);
            v2=torch.zeros(100).cuda()+1e-7;
            v2[box['category_id']]=1;
            fvs_pool.append(v.view(1,-1)*v2.view(-1,1));
    
    fvs_pool=torch.stack(fvs_pool,dim=0);
    fvs_pool=fvs_pool.mean(dim=0);
    
    return fvs_pool;


def pool3(fvs):
    #Return a GTboxes by 100 matrix
    fvs_pool=[];
    for imid,data in enumerate(fvs):
        pred=data['pred']
        ann=data['annotation']
        for boxid,box in enumerate(ann):
            v=torch.zeros(100)+1e-7;
            v2=torch.zeros(100)+1e-7;
            v2[box['category_id']]=1;
            for boxid2,box2 in enumerate(pred['boxes'].tolist()):
                s=iou(box['bbox'],box2)
                if s>0:
                    score=float(pred['scores'][boxid2]);
                    label=int(pred['labels'][boxid2]);
                    v[label]=max(v[label],score*s)
            
            v=F.normalize(v,dim=0,p=1);
            fvs_pool.append(torch.cat((v,v2),dim=0));
    
    
    fvs_pool=torch.stack(fvs_pool,dim=0)
    
    v={};
    for i in range(fvs_pool.shape[0]):
        c=int(fvs_pool[i,100:].eq(1).nonzero())
        if not c in v:
            v[c]=[];
        
        v[c].append(fvs_pool[i,:100]);
    
    v2=torch.eye(100);
    for c in v:
        v2[c,:]=sum(v[c])/len(v[c])
    
    return v2;


def iou(box1,box2,eps=1e-7):
    x0,y0,w0,h0=box1;
    x1,y1,w1,h1=box2;
    
    overlapx=max(min(x0+w0,x1+w1)-max(x0,x1),0)
    overlapy=max(min(y0+h0,y1+h1)-max(y0,y1),0)
    
    i=overlapx*overlapy;
    s=i/(abs(w0*h0+w1*h1-i)+eps)
    return s;




if __name__ == "__main__":
    data=db.Table({'model_id':[],'model_name':[],'fvs':[]}); #,'label':[] label currently incorrect
    data=db.DB({'table_ann':data});
    
    t0=time.time()
    
    default_params=smartparse.obj();
    default_params.fname='data_r10_pred.pt'
    params=smartparse.parse(default_params);
    params.argv=sys.argv;
    data.d['params']=db.Table.from_rows([vars(params)]);
    
    model_ids=list(range(0,144))
    
    for i,id in enumerate(model_ids):
        print(i,id)
        fv=extract_fv(id,params=params);
        
        #Load GT
        fname=os.path.join(helper.get_root(id),'ground_truth.csv');
        f=open(fname,'r');
        for line in f:
            line.rstrip('\n').rstrip('\r')
            label=int(line);
            break;
        
        f.close();
        
        data['table_ann']['model_name'].append('id-%08d'%id);
        data['table_ann']['model_id'].append(id);
        #data['table_ann']['label'].append(label);
        data['table_ann']['fvs'].append(fv);
        
        print('Model %d(%d), time %f'%(i,id,time.time()-t0));
        if i%1==0:
            data.save(params.fname);
    
    data.save(params.fname);
