import torch
import torchvision.models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import math
from collections import OrderedDict as OrderedDict
import copy


class MLP(nn.Module):
    def __init__(self,ninput,nh,noutput,nlayers):
        super().__init__()
        
        self.layers=nn.ModuleList();
        self.bn=nn.LayerNorm(ninput);
        if nlayers==1:
            self.layers.append(nn.Linear(ninput,noutput));
        else:
            self.layers.append(nn.Linear(ninput,nh));
            for i in range(nlayers-2):
                self.layers.append(nn.Linear(nh,nh));
            
            self.layers.append(nn.Linear(nh,noutput));
        
        self.ninput=ninput;
        self.noutput=noutput;
        return;
    
    def forward(self,x):
        h=x.view(-1,self.ninput);
        #h=self.bn(h);
        for i in range(len(self.layers)-1):
            h=self.layers[i](h);
            h=F.relu(h);
            #h=F.dropout(h,training=self.training);
        
        h=self.layers[-1](h).view(*(list(x.shape[:-1])+[-1]));
        return h



class vector_log(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()*torch.log1p(x.abs())/10
    
    @staticmethod
    def backward(ctx, grad_output):
        x=ctx.saved_tensors
        return grad_output/(1+x.abs())/10

vector_log = vector_log.apply


def fv_pile(hx,hx2,hxy):
    L,N=hx.shape
    hxm=hx.mean(dim=0)
    hx2m=hx2.mean(dim=0)
    hxym=hxy.mean(dim=0)
    
    fv_aaa=[] #N3
    fv_aaa.append(hxm)
    fv_aaa.append((hx**2).mean(dim=0))
    fv_aaa.append(hx2m)
    fv_aaa=torch.stack(fv_aaa,dim=-1)
    
    fv_abb=[] #N3
    fv_abb.append(hxm)
    fv_abb.append((hx**2).mean(dim=0))
    fv_abb.append(hx2m)
    fv_abb=torch.stack(fv_abb,dim=-1)
    
    fv_aab=[] #NN3
    fv_aab.append(hxm.view(-1,1)*hxm.view(1,-1))
    fv_aab.append((hx.view(L,N,1)*hx.view(L,1,N)).mean(dim=0))
    fv_aab.append(hxym)
    fv_aab=torch.stack(fv_aab,dim=-1)
    
    fv_aba=[] #NN3
    fv_aba.append(hxm.view(-1,1)*hxm.view(1,-1))
    fv_aba.append((hx.view(L,N,1)*hx.view(L,1,N)).mean(dim=0))
    fv_aba.append(hxym)
    fv_aba=torch.stack(fv_aba,dim=-1)
    
    fv_abc=[] #NN3
    fv_abc.append(hxm.view(-1,1)*hxm.view(1,-1))
    fv_abc.append((hx.view(L,N,1)*hx.view(L,1,N)).mean(dim=0))
    fv_abc.append(hxym)
    fv_abc=torch.stack(fv_abc,dim=-1)
    
    fv_aaa,fv_abb,fv_aab,fv_aba,fv_abc=[vector_log(fv*1e3) for fv in (fv_aaa,fv_abb,fv_aab,fv_aba,fv_abc)]
    
    return fv_aaa,fv_abb,fv_aab,fv_aba,fv_abc


class new(nn.Module):
    def __init__(self,params):
        super(new,self).__init__()
        
        
        nh=params.nh;
        nh2=params.nh2;
        nh3=params.nh3;
        nlayers=params.nlayers
        nlayers2=params.nlayers2
        nlayers3=params.nlayers3
        self.margin=params.margin
        
        self.encoder1=MLP(3,nh,nh,nlayers)
        self.encoder2=MLP(3,nh,nh,nlayers)
        self.encoder3=MLP(3,nh,nh,nlayers)
        self.encoder4=MLP(3,nh,nh,nlayers)
        self.encoder5=MLP(3,nh,nh,nlayers)
        
        
        self.encoder6=MLP(nh,nh2,nh2,nlayers2);
        self.encoder7=MLP(nh2,nh3,2,nlayers3);
        
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        
        return;
    
    def forward(self,data_batch):
        h=[];
        for i in range(len(data_batch['hx'])):
            hx=data_batch['hx'][i].cuda()
            hx2=data_batch['hx2'][i].cuda()
            hxy=data_batch['hxy'][i].cuda()
            fv_aaa,fv_abb,fv_aab,fv_aba,fv_abc=fv_pile(hx,hx2,hxy)
            
            fv_aaa=self.encoder1(fv_aaa)
            fv_abb=self.encoder2(fv_abb).mean(dim=0)
            fv_aab=self.encoder3(fv_aab).mean(dim=1)
            fv_aba=self.encoder3(fv_aba).mean(dim=0)
            fv_abc=self.encoder3(fv_abc).mean(dim=(0,1))
            
            fv=fv_aaa+fv_abb+fv_aab+fv_aba+fv_abc
            fv=self.encoder6(fv).mean(dim=0)
            h.append(fv)
        
        h=torch.stack(h,dim=0)
        h=self.encoder7(h)
        
        h=torch.tanh(h)*self.margin;
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];
    
