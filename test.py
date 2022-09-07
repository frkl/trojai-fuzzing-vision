import torch
import helper_r10 as helper
import engine_objdet as engine

#First try to run inference on a model
id=5;
model_filepath, scratch_dirpath, examples_dirpath=helper.get_paths(id);
trigger=helper.load_trigger(id);
print(trigger.shape)

interface=engine.new(model_filepath);
#examples=interface.load_examples('/work2/project/trojai-datasets/coco2017/trojai_coco',bsz=100) #examples_dirpath
examples=interface.load_examples('/work2/project/trojai-datasets/round10-train-dataset/models/id-00000005/poisoned-example-data',bsz=100) #examples_dirpath
bbox=torch.Tensor([0.1,0.5,0.2,0.2]);
triggered_examples=interface.insert_trigger(examples,trigger,bbox.view(1,4).repeat(len(examples),1));

#helper.visualize(triggered_examples['im'][0],out='tmp.png')
#a=0/0

results=interface.inference(examples);
results2=interface.inference(triggered_examples);

before=torch.cat([r['logits'] for r in results],dim=0)

after=torch.cat([r['logits'] for r in results2],dim=0)


def solve(before,after,reg=0.0):
    before=before.double();
    after=after.double();
    before=torch.cat((before,torch.ones(before.shape[0],1).to(before.device)),dim=1)
    A=torch.mm(before.t(),before)+reg*torch.eye(before.shape[-1]).to(before.device);
    T=torch.mm(before.t(),after);
    W=torch.mm(A.inverse(),T);
    return W;


W=solve(before.double(),after.double(),reg=0.01)





results2=interface.inference(triggered_examples);
a=0/0
err=torch.stack(([r['loss']['bbox_regression'] for r in results]),dim=0)
err2=torch.stack(([r['loss']['bbox_regression'] for r in results2]),dim=0)

for i in range(len(examples)):
    helper.visualize(examples['imname'][i],results[i]['pred'],'vis/clean%d.png'%i)
    #helper.visualize(triggered_examples['im'][i],results2[i]['pred'],'vis/trigger%d.png'%i)




import weight_analysis 

fvs=weight_analysis.run(interface)




import cv2
import os
import numpy as np
import random
import json
import copy
import torch
import torch.nn.functional as F
import torchvision
import util.db as db
import math

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



def pool4(fvs,th=0.5):
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
                if s>th:
                    score=float(pred['scores'][boxid2]);
                    label=int(pred['labels'][boxid2]);
                    v[label]=max(v[label],score)
            
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


examples_dirpath='/work2/project/trojai-datasets/coco2017/trojai_coco'

fns=[os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.jpg')]
fns.sort()

fnames=[fn for fn in os.listdir(examples_dirpath) if fn.endswith('.jpg')]
fnames.sort()
fnames=fnames[::5];
for f in fnames:
    fin=os.path.join(examples_dirpath,f);
    os.system('cp %s ./sample_images_1k/'%fin)
    os.system('cp %s.json ./sample_images_1k/'%fin[:-4]);

import torchvision
from torchvision import transforms
import torchvision.datasets.folder
def saveim(fv,fname):
    fv=fv.repeat(3,1,1)/fv.max()
    torchvision.utils.save_image(fv,fname);


import torch
import trigger_search
data=torch.load('data_r10_pred_1k.pt');
#fvs=[trigger_search.pool(fv) for fv in data['table_ann']['fvs']];

#Visualize fvs in heatmap
fvs=[];
for i,fv in enumerate(data['table_ann']['fvs']):
    print(i)
    fv0=pool4(fv,0.25)
    fv1=pool4(fv,0.5)
    fv2=pool4(fv,0.75)
    fv0=torch.stack((fv0,fv1,fv2),dim=0);
    fvs.append(fv0);

fvs=torch.stack(fvs,dim=0);

data['table_ann']['fvs']=fvs;


torch.save(data,'data_r10_pred_1k_pool5.pt')

saveim(fvs[1],'tmp.png')

