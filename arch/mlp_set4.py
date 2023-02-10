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
        return h





#Input n x K x ninput
#Output n x nh
class encoder(nn.Module):
    def __init__(self,ninput,K,nh,nlayers):
        super().__init__()
        #ninput => K*nh
        #ninput => K*ninput
        if nlayers>=1:
            self.net=MLP(ninput,nh,nh,nlayers);
        else:
            self.net=nn.Identity();
        
        self.q=torch.arange(0,K)/(K-1);
        if nlayers>=1:
            self.nh=K*nh
        else:
            self.nh=K*ninput
            
    
    
    def forward(self,x):
        N,ninput=x.shape
        h=self.net(x)
        h=torch.quantile(h,self.q.to(h.device),dim=0,keepdim=True)
        #print(h.max(),h.min())
        return h.contiguous().view(-1)


class new(nn.Module):
    def __init__(self,params):
        super(new,self).__init__()
        
        
        nh=params.nh;
        nh2=params.nh2;
        nh3=params.nh3;
        nlayers=params.nlayers-1
        nlayers2=params.nlayers2-1
        nlayers3=params.nlayers3
        
        Q=10;
        
        #self.enc_poly_grads=encoder(10,Q,nh,nlayers)
        #self.enc_filter_grads=encoder(20,Q,nh,nlayers)
        
        self.enc_poly_preds=encoder(200,Q,nh,nlayers)
        self.enc_filter_preds=encoder(400,Q,nh2,nlayers2)
        
        nh_combined=sum([net.nh for net in [self.enc_poly_preds,self.enc_filter_preds]]);
        
        self.encoder_combined=MLP(nh_combined,nh3,2,nlayers3);
        
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        
        return;
    
    def forward(self,data_batch):
        h=[];
        #Have to process one by one due to variable nim & nclasses
        for i in range(len(data_batch['poly_grads'])):
            
            #poly_grads=F.normalize(data_batch['poly_grads'][i].view(10,-1),dim=-1).permute(1,0)
            #filter_grads=F.normalize(data_batch['filter_grads'][i].view(20,-1),dim=-1).permute(1,0)
            #poly_preds=data_batch['poly_preds_after'][i].view(200,-1).permute(1,0)
            #filter_preds=data_batch['filter_preds_after'][i][10:].contiguous().view(400,-1).permute(1,0)
            poly_preds=torch.sigmoid(data_batch['poly_preds_after'][i].view(200,-1).permute(1,0))
            filter_preds=torch.sigmoid(data_batch['filter_preds_after'][i][:].contiguous().view(400,-1).permute(1,0))
            
            h_i=[self.enc_poly_preds(poly_preds),self.enc_filter_preds(filter_preds)]
            h_i=torch.cat(h_i,dim=0);
            h.append(h_i);
        
        h=torch.stack(h,dim=0);
        h=self.encoder_combined(h);
        h=torch.tanh(h)*8;
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];
    