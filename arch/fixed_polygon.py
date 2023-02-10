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
    # x y w h: location of polygon
    # C H W: whole image shape
    def __init__(self,C,h,w):
        super(new,self).__init__()
        self.trigger=nn.Parameter(torch.Tensor(1,C,h,w).uniform_(-0.1,0.1));
        self.mask=nn.Parameter(torch.Tensor(1,1,h,w).uniform_(-0.1,0.1)+0.2);
        
        return;
    
    def forward(self,I,x=16,y=16,opacity=0.9,s=1.0,n=1):
        _,C,h,w=self.trigger.shape
        N,_,H,W=I.shape
        trigger=F.sigmoid(self.trigger)
        mask=F.sigmoid(self.mask);
        '''
        transform=torch.Tensor(N,2,3).fill_(0)
        transform[:,0,0]=5;
        transform[:,1,1]=5;
        transform[:,:,2]=torch.Tensor(N,2).uniform_(-0.9,0.9)*5;
        
        grid=F.affine_grid(transform,I.size()).to(mask.device);
        mask_=F.grid_sample(mask.repeat(N,1,1,1),grid);
        trigger_=F.grid_sample(trigger.repeat(N,1,1,1),grid);
        '''
        
        out=[];
        for _ in range(n):
            y=int(torch.LongTensor(1).random_(H-h))
            x=int(torch.LongTensor(1).random_(W-w))
            
            
            trigger_=F.pad(trigger,(x,W-x-w,y,H-y-h));
            mask_=F.pad(mask,(x,W-x-w,y,H-y-h));
            
            if torch.LongTensor(1).random_(2)>0:
                x=torch.arange(-2,2,0.124).view(1,-1)
                px=torch.exp(-x**2/2);
                y=torch.arange(-2,2,0.124).view(-1,1)
                py=torch.exp(-y**2/2);
                k=px*py;
                k=k/k.sum();
                
                blur_kernel=torch.stack((k,k*0,k*0,k*0,k,k*0,k*0,k*0,k),dim=0).view(3,3,k.shape[0],k.shape[1]).to(trigger_.device);
                
                trigger_=F.conv2d(trigger_,blur_kernel,padding=(k.shape[0]-1)//2)
            
            out_=I*(1-mask_)+trigger_*mask_
            
            out.append(out_)
        
        out=torch.cat(out,dim=0);
        return out#,v*s,v;
    
