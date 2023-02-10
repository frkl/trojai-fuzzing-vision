import torch
import torchvision.models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import math
from collections import OrderedDict as OrderedDict
import copy

class new(nn.Module):
    def __init__(self,h,w,alpha_bias=-0.1,m=10):
        super(new,self).__init__()
        self.alpha=nn.Parameter(torch.Tensor(1,1,h,w).uniform_(-0.01,0.01)+alpha_bias)
        self.trigger=nn.Parameter(torch.Tensor(1,3,h,w).uniform_(-0.01,0.01))
        
        self.s=nn.Parameter(torch.Tensor(1).fill_(0))
        self.theta=nn.Parameter(torch.Tensor(1).fill_(0))
        self.offset=nn.Parameter(torch.Tensor(2).fill_(0))
        
        self.m=m;
        return;
    
    def forward(self,I,aug=True):
        N=I.shape[0];
        trigger=torch.sigmoid(self.trigger*self.m);
        alpha=torch.sigmoid(self.alpha*self.m);
        
        if aug:
            #Randomly perturb the trigger using affine transforms so we find robust triggers and not adversarial noise
            transform=[];
            for _ in range(N):
                #scale
                sz=torch.exp(torch.Tensor(1).uniform_(-0.2,0.2).to(alpha.device)-self.s);
                #smol rotation
                theta=torch.Tensor(1).uniform_(-3.14/6,3.14/6).to(alpha.device)+self.theta;
                #5% imsz offset
                pad=0.3;
                offset=torch.Tensor(2).uniform_(-pad,pad).to(alpha.device)+self.offset;
                t=torch.stack([sz*torch.cos(theta),-sz*torch.sin(theta),offset[0:1],sz*torch.sin(theta),sz*torch.cos(theta),offset[1:2]],dim=0).view(2,3);
                transform.append(t);
            
            transform=torch.stack(transform,dim=0)
            grid=F.affine_grid(transform,I.size()).to(alpha.device);
            #Synthesize trigger
            alpha=F.grid_sample(alpha.repeat(N,1,1,1),grid);
            trigger=F.grid_sample(trigger.repeat(N,1,1,1),grid);
        
        out=(1-alpha)*I+alpha*trigger;
        return out;
    
    def characterize(self):
        alpha=torch.sigmoid(self.alpha*self.m)*torch.exp(self.s)
        l2=alpha.mean() #portion of the original image with trigger coverage
        
        alpha_6=torch.sigmoid(self.alpha*self.m*2+6)*torch.exp(self.s)
        l2_salient=alpha_6.mean() #portion of the original image with significant trigger coverage
        
        _,_,h,w=self.alpha.shape
        wt=torch.Tensor(list(range(w)))/w-0.5;
        wt=wt.view(1,1,1,w).to(alpha.device);
        ht=torch.Tensor(list(range(h)))/h-0.5;
        ht=ht.view(1,1,h,1).to(alpha.device);
        
        dist=alpha/(alpha.sum()+1e-8);
        Ew=(dist*wt).sum();
        Eh=(dist*ht).sum()
        Ew2=(dist*(wt**2)).sum();
        Eh2=(dist*(ht**2)).sum();
        varw=Ew2-Ew**2;
        varh=Eh2-Eh**2;
        stdw=torch.sqrt(varw.clamp(min=1e-12));
        stdh=torch.sqrt(varh.clamp(min=1e-12));
        
        fv=torch.stack((l2,l2_salient,Eh,Ew,stdh,stdw),dim=0)*torch.exp(self.s);
        
        return fv.view(-1); #6-dim
    
    def complexity(self):
        fv=self.characterize();
        return fv[-1]+fv[-2]+fv[0]+fv[1]
    
    def diff(self,other):
        trigger=torch.sigmoid(self.trigger*self.m);
        alpha=torch.sigmoid(self.alpha*self.m);
        trigger=F.normalize((trigger*alpha).view(-1),dim=0)
        
        trigger_2=torch.sigmoid(other.trigger.data*other.m);
        alpha_2=torch.sigmoid(other.alpha.data*other.m);
        trigger_2=F.normalize((trigger_2*alpha_2).view(-1),dim=0)
        
        diff=(trigger*trigger_2).sum(); 
        return diff
    