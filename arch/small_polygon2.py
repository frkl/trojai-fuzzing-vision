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
    def __init__(self,c,h,w,nim,alpha_bias=0.5,m=10):
        super(new,self).__init__()
        self.mask=nn.Parameter(torch.Tensor(1,1,h,w).uniform_(-0.1+alpha_bias,0.1+alpha_bias))
        self.trigger=nn.Parameter(torch.Tensor(1,c,h,w).uniform_(-0.1,0.1))
        
        self.s=-2.5 #nn.Parameter(torch.Tensor(1).fill_(0))
        
        self.sz=nn.Parameter(torch.Tensor(nim).fill_(1))
        self.offset=nn.Parameter(torch.Tensor(nim,2).uniform_(-1,1))
        
        self.h=h;
        self.w=w;
        return;
    
    def forward(self,I,n=1,aug=True,intensity=1):
        N=I.shape[0];
        
        out=[];
        for i in range(n):
            trigger=torch.sigmoid(self.trigger);
            mask=torch.sigmoid(self.mask);
            
            if aug:
                #Randomly perturb the trigger using affine transforms so we find robust triggers and not adversarial noise
                transform=[];
                for j in range(N):
                    #scale
                    scale=max(self.h,self.w)/min(I.shape[-1],I.shape[-2])
                    sz=torch.exp(-self.sz[j:j+1])/scale; 
                    offset=torch.tanh(self.offset[j])
                    offset=offset*sz
                    t=torch.stack([sz,sz*0,offset[0:1],sz*0,sz,offset[1:2]],dim=0).view(2,3);
                    transform.append(t);
                
                transform=torch.stack(transform,dim=0)
                grid=F.affine_grid(transform,I.size()).to(mask.device);
                #Synthesize trigger
                mask=F.grid_sample(mask.repeat(N,1,1,1),grid);
                trigger=F.grid_sample(trigger.repeat(N,1,1,1),grid);
            
            out_=I*(1-mask)+trigger*mask;
            out.append(out_)
        
        out=torch.cat(out,dim=0).clamp(max=1,min=0);
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
    
