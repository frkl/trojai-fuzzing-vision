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
        
        h=self.layers[-1](h);
        h=h.view(*(list(x.shape[:-1])+[-1]));
        return h


class new(nn.Module):
    def __init__(self,params):
        super(new,self).__init__()
        nh=params.nh;
        nh3=params.nh3;
        nlayers=params.nlayers
        nlayers2=params.nlayers2
        
        q=int((params.nh2//2)**0.5);
        self.q=torch.arange(0,1+1e-8,1/q).cuda()
        q=len(self.q);
        
        try:
            self.margin=params.margin
        except:
            self.margin=8;
        
        self.encoder_hist=MLP(10000,nh,2,nlayers);
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        return;
    
    def forward(self,data_batch):
        weight_dist=data_batch['fvs'];
        b=len(weight_dist);
        
        h=self.encoder_hist(weight_dist.cuda().view(b,-1));
        h=torch.tanh(h)*self.margin;
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];
    