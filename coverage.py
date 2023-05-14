import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim

import os
import sys
import time
import importlib
import torch
import copy
import torchvision
from torchvision import transforms
import torchvision.datasets.folder
from pathlib import Path

from torch.autograd import grad

import util.db as db
import util.smartparse as smartparse
import util.session_manager as session_manager

from util.session_manager import loss_tracker
import helper_r13_v2 as helper


class diversity_grad:
    def __init__(self,interface,kw=None):
        self.interface=interface
        self.kw=kw
        return;
    
    def embed(self,data):
        loss=self.interface.eval_loss(data)
        params=[]
        names=[];
        for x,y in self.interface.model.named_parameters():
            if self.kw is None or x.find(self.kw)>=0:
                params.append(y)
                names.append(x)
        
        g=grad(loss,list(params),create_graph=True, allow_unused=True);
        e={};
        for i in range(len(names)):
            if g[i] is None:
                e[names[i]]=(params[i]*0).contiguous().data
            else:
                e[names[i]]=g[i].contiguous().data;
        
        return e
    
    def sim_list(self,e1,e2):
        names=list(e1[0].keys());
        e1=torch.stack([torch.cat([e1[i][x].view(-1) for x in names],dim=0) for i in range(len(e1))],dim=0);
        e2=torch.stack([torch.cat([e2[i][x].view(-1) for x in names],dim=0) for i in range(len(e2))],dim=0);
        e1=F.normalize(e1,dim=-1)
        e2=F.normalize(e2,dim=-1)
        s=torch.mm(e1,e2.t())
        return s
        
    
    def coverage(self,es1,es2):
        s=self.sim_list(es1,es2)
        s=s.max(dim=-1)[0].mean();
        return s


def coverage(diversity,e1,e2):
    e1=[diversity.embed(x) for x in e1]
    e2=[diversity.embed(x) for x in e2]
    sim=diversity.coverage(e1,e2)
    return sim


default_params=smartparse.obj();
default_params.session_dir=None;
default_params.div='cm.grad';
default_params.model=2;

default_params.arch='arch.meta_poly7';
default_params.opt='conv_smp';
default_params.load='';
default_params.round=90;
default_params.iter=2000;
default_params.lr=3e-4;
params=smartparse.parse(default_params)
params.argv=sys.argv
#session=session_manager.create_session(params);


diversity=importlib.import_module(params.div)


models=list(range(0,128))
models=['id-%08d'%i for i in models];
models=[os.path.join(helper.root(),'models',x) for x in models];

for model_id in [2,24,31,33,35,43,46,49,51,64,73,74,91,98,107,111,114,115,117,118,127]:
    interface=helper.engine(folder=models[model_id]) #2 7 11
    
    examples=interface.load_examples()
    poisoned_examples=interface.load_poisoned_examples()
    clean_source_examples=interface.clean_source_examples(n=len(poisoned_examples))
    clean_target_examples=interface.clean_target_examples(n=len(poisoned_examples))
    cleansed_source_examples=interface.cleansed_source_examples()
    cleansed_target_examples=interface.cleansed_target_examples()
    
    
    
    for i,ex in enumerate([poisoned_examples,clean_source_examples,clean_target_examples,cleansed_source_examples,cleansed_target_examples]):
        pred=interface.eval(ex[1])
        #print(clean_poisoned_examples[0]['gt'])
        Path('vis/model_%d/'%model_id).mkdir(parents=True, exist_ok=True)
        helper.visualize(ex[1]['image'][0],pred,out='vis/model_%d/%d.png'%(model_id,i),threshold=0.1)

    names=[name for name,param in interface.model.named_parameters()]


    examples=list(examples.rows())
    poisoned_examples=list(poisoned_examples.rows())
    cleansed_source_examples=list(cleansed_source_examples.rows())
    cleansed_target_examples=list(cleansed_target_examples.rows())


    n=len(poisoned_examples);
    poisoned_a=poisoned_examples[:n//2]
    poisoned_b=poisoned_examples[n//2:]
    
    examples=examples[:n//2]
    clean_source=clean_source_examples[:n//2]
    clean_target=clean_target_examples[:n//2]
    cleansed_source=cleansed_source_examples[:n//2]
    cleansed_target=cleansed_target_examples[:n//2]
    
    '''
    experiment_set=[examples]
    '''
    experiment_set=[examples,poisoned_a,poisoned_b,clean_source,cleansed_source,clean_target,cleansed_target]
    experiment_set.append(interface.target_to_source(poisoned_a))
    experiment_set.append(interface.target_to_source(poisoned_b))
    experiment_set.append(interface.source_to_target(clean_source))
    experiment_set.append(interface.source_to_target(cleansed_source))
    experiment_set.append(interface.target_to_source(clean_target))
    experiment_set.append(interface.target_to_source(cleansed_target))
    
    print(interface.model.__class__.__name__)
    print(names)
    if interface.model.__class__.__name__=='SSD':
        kws=['backbone','head.classification_head','head.regression_head',None]
    elif interface.model.__class__.__name__=='DetrForObjectDetection':
        kws=['class_labels_classifier','bbox_predictor','model.encoder','model.decoder',None]
    else:
        kws=['backbone.body','backbone.fpn','rpn.head','roi_heads.box_head','roi_heads.box_predictor',None]



    N=len(experiment_set)
    
    scores=torch.Tensor(len(kws),N,N).fill_(0);
    t0=time.time();
    for k,kw in enumerate(kws):
        d=diversity.new(interface,kw=kw)
        for i in range(N):
            for j in range(N):
                print('%d/%d, time %.2f'%(k*N*N+i*N+j,len(kws)*N*N,time.time()-t0))
                scores[k,i,j]=coverage(d,experiment_set[i],experiment_set[j]);
    
    torch.save(scores,'scores_%s_%s_%d.pt'%(params.div,interface.model.__class__.__name__,model_id));


a=0/0

poison a [1,7]
poison b [2,8]

cleansed source [4,10]


expind=[8,2,4,10,6,12,9,5,0]

def save(tensor,fname='tmp.csv'):
    n,m=tensor.shape
    f=open(fname,'w');
    for i in range(n):
        for j in range(m):
            f.write('%.3f,'%tensor[i,j])
        
        f.write('\n')
    
    f.close()


all layers


s_poisoned          = coverage(d,poisoned_b,poisoned_a)
s_clean             = coverage(d,poisoned_b,examples[:len(poisoned_a)]) 
s_source            = coverage(d,poisoned_b,cleansed_source_examples[:len(poisoned_a)]) 
s_target            = coverage(d,poisoned_b,cleansed_target_examples[:len(poisoned_a)]) 

print(s_poisoned,s_clean,s_source,s_target)


s_poisoned          = coverage(d,poisoned_b2,poisoned_a2)
s_clean             = coverage(d,poisoned_b2,examples[:len(poisoned_a)]) 
s_source            = coverage(d,poisoned_b2,cleansed_source_examples[:len(poisoned_a)]) 
s_target            = coverage(d,poisoned_b2,cleansed_target_examples[:len(poisoned_a)]) 

print(s_poisoned,s_clean,s_source,s_target)

a=0/0

a=0/0


loss=interface.eval_loss(examples[0])

#2 61=>44
#7 9=>23

for x in poisoned_examples:
    for box in x['gt']:
        if box['label']==23:
            box['label']=9;




diversity=diversity.new()

g=[diversity.compute_grad(interface,x['image'],x['gt']).data.clone() for x in poisoned_examples]
g=torch.stack(g,dim=0)
gn=F.normalize(g,dim=-1)
div=torch.mm(gn,gn.t())


for i,x in enumerate(poisoned_examples):
    im=x['image']
    data=interface.eval({'image':x['image']})
    
    torchvision.utils.save_image(im,session.file('vis','%d.png'%i))
    helper.visualize(session.file('vis','%d.png'%i),data,session.file('vis_ann','%d.png'%i))




