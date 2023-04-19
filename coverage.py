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

from torch.autograd import grad

import util.db as db
import util.smartparse as smartparse
import util.session_manager as session_manager

from util.session_manager import loss_tracker
import helper_r13_v2 as helper



class diversity_grad:
    def __init__(self):
        return;
    
    def compute_grad(self,interface,im,gt):
        loss=interface.eval_loss({'image':im,'gt':gt})
        g=grad(loss,list(interface.model.parameters()),create_graph=True, allow_unused=True);
        g=[x.contiguous().view(-1) for x in g if not x is None]
        g=torch.cat(g,dim=0);
        return g
    
    def sim_utilization(self,g0,g1):
        g0=F.normalize(g0.view(-1),dim=-1)
        g1=F.normalize(g1.view(-1),dim=-1)
        diff=(g0*g1).sum()
        return diff
    
    def loss_util(self,interface,im,gt,target_util):
        if len(target_util)==0:
            return im.sum()*0
        
        g=self.compute_grad(interface,im,gt)
        gdiff=[self.sim_utilization(g,g2) for g2 in target_util]
        gdiff=torch.stack(gdiff,dim=0).max();
        #print(gdiff)
        return gdiff
    
    def util(self,x,interface,im0,gt):
        im=x.apply(im0)
        return self.compute_grad(interface,im,gt)
    
    def loss(self,x,interface,im0,gt,target_util):
        im=x.apply(im0)
        return self.loss_util(interface,im,gt,target_util)




default_params=smartparse.obj();
default_params.session_dir=None;
default_params.arch='arch.meta_poly7';
default_params.opt='conv_smp';
default_params.div='diversity_grad';
default_params.load='';
default_params.round=90;
default_params.iter=2000;
default_params.lr=3e-4;
params=smartparse.parse(default_params)
params.argv=sys.argv
session=session_manager.create_session(params);




models=list(range(0,20))#[5,6,7,8,9]
models=['id-%08d'%i for i in models];
models=[os.path.join(helper.root(),'models',x) for x in models];

interface=helper.engine(folder=models[7]) #2 7 11
examples=interface.load_examples()
poisoned_examples=interface.load_poisoned_examples()

loss=interface.eval_loss(examples[0])

#2 61=>44
#7 9=>23

for x in poisoned_examples:
    for box in x['gt']:
        if box['label']==23:
            box['label']=9;




diversity=diversity_grad()

g=[diversity.compute_grad(interface,x['image'],x['gt']).data.clone() for x in poisoned_examples]
g=torch.stack(g,dim=0)
gn=F.normalize(g,dim=-1)
div=torch.mm(gn,gn.t())


for i,x in enumerate(poisoned_examples):
    im=x['image']
    data=interface.eval({'image':x['image']})
    
    torchvision.utils.save_image(im,session.file('vis','%d.png'%i))
    helper.visualize(session.file('vis','%d.png'%i),data,session.file('vis_ann','%d.png'%i))




