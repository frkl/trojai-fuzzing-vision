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

#BhXY => Bh10
class inv23(nn.Module):
    def forward(self,x):
        h120=x.sum(dim=(-1,-2))
        
        h220=(x*x).sum(dim=(-1,-2))# aa bb
        h230=(x.sum(dim=-1)*x.sum(dim=-1)).sum(dim=-1)# aa bc
        h231=(x.sum(dim=-2)*x.sum(dim=-2)).sum(dim=-1)# ab cc
        
        h320=(x*x*x).sum(dim=(-1,-2)) # aaa bbb
        h330=((x*x).sum(dim=-1)*x.sum(dim=-1)).sum(dim=-1) # aaa bbc
        h331=((x*x).sum(dim=-2)*x.sum(dim=-2)).sum(dim=-1) # aab ccc
        h340=(x.sum(dim=-1)*x.sum(dim=-1)*x.sum(dim=-1)).sum(dim=-1)# aaa bcd
        h341=(x.sum(dim=-2)*x.sum(dim=-2)*x.sum(dim=-2)).sum(dim=-1)# abc ddd
        h342=(torch.bmm(x.sum(dim=-1,keepdim=True).transpose(-1,-2),x).squeeze(dim=-2)*x.sum(dim=-2)).sum(dim=-1)# aab cdd
        return torch.stack((h120,h220,h230,h231,h320,h330,h331,h340,h341,h342),dim=-1) #bh 10


#BhXY => Bh10
class inv_ii_3(nn.Module):
    def forward(self,x):
        h120=x.sum(dim=(-1,-2))
        
        h220=(x*x).sum(dim=(-1,-2))# aa bb
        h230=(x.sum(dim=-1)*x.sum(dim=-1)).sum(dim=-1)# aa bc
        h231=(x.sum(dim=-2)*x.sum(dim=-2)).sum(dim=-1)# ab cc
        
        h320=(x*x*x).sum(dim=(-1,-2)) # aaa bbb
        h330=((x*x).sum(dim=-1)*x.sum(dim=-1)).sum(dim=-1) # aaa bbc
        h331=((x*x).sum(dim=-2)*x.sum(dim=-2)).sum(dim=-1) # aab ccc
        h340=(x.sum(dim=-1)*x.sum(dim=-1)*x.sum(dim=-1)).sum(dim=-1)# aaa bcd
        h341=(x.sum(dim=-2)*x.sum(dim=-2)*x.sum(dim=-2)).sum(dim=-1)# abc ddd
        h342=(torch.bmm(x.sum(dim=-1,keepdim=True).transpose(-1,-2),x).squeeze(dim=-2)*x.sum(dim=-2)).sum(dim=-1)# aab cdd
        return torch.stack((h120,h220,h230,h231,h320,h330,h331,h340,h341,h342),dim=-1) #bh 10


#BhX => Bh10
class inv13(nn.Module):
    def forward(self,x):
        h1=x.sum(dim=-1)
        h2=(x**2).sum(dim=-1)
        h3=(x**3).sum(dim=-1)
        
        return torch.stack((h1,h2,h3),dim=-1) #bh 10



class encoder(nn.Module):
    def __init__(self,ninput,nh,nlayers):
        super().__init__()
        self.layers=nn.ModuleList();
        for i in range(nlayers):
            if i==0:
                self.layers.append(nn.Conv2d(ninput,nh,5,padding=2));
            else:
                self.layers.append(nn.Conv2d(nh,nh,5,padding=2));
        
    
    
    def forward(self,x):
        h=x;
        for i,layer in enumerate(self.layers):
            if i>0:
                h=F.relu(h);
                h=h+layer(h)
            else:
                h=layer(h)
        
        h=h.mean(dim=-1).mean(dim=-1);
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
        
        self.pool=inv_ii_3()
        self.t=nn.ModuleList()
        for i in range(6)
        self.encoder1=MLP(108,nh,nh2,nlayers)
        self.encoder3=MLP(3*nh2,nh3,2,nlayers3);
        
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        
        return;
    
    def process_fvs(self,idx,val,id=0):
        val=torch.Tensor(val,device=self.w.device)
        idx=torch.LongTensor(val,device=self.w.device)
        val=self.t(val)
        
        M=torch.sparse_coo_tensor(idx.t(),val).coalesce().to_dense() #xyh
        h=self.pool(M.permute(2,0,1)) #h-10
        return h.view(-1)
    
    
    def forward(self,data_batch):
        with torch.no_grad():
            #print([fv.shape for fv in data_batch['fvs']]);
            fvs=[fv.to(self.w.device).view(-1,fv.shape[-1]) for fv in data_batch['fvs']];
        
        bsz=len(data_batch['idx0'])
        h=[]
        for i in range(bsz):
            ntrig=len(data_batch['idx0'][i])
            h_i=[]
            for j in range(ntrig):
                h_j=[]
                h_j.append(self.process_fvs(data_batch['idx0'][i][j],data_batch['val0'][i][j]))
                h_j.append(self.process_fvs(data_batch['idx1'][i][j],data_batch['val1'][i][j]))
                h_j.append(self.process_fvs(data_batch['idx2'][i][j],data_batch['val2'][i][j]))
                h_j.append(self.process_fvs(data_batch['idx3'][i][j],data_batch['val3'][i][j]))
                h_j.append(self.process_fvs(data_batch['idx4'][i][j],data_batch['val4'][i][j]))
                h_j.append(self.process_fvs(data_batch['idx5'][i][j],data_batch['val5'][i][j]))
                h_j=torch.cat(h_j,dim=0)
                h_i.append(h_j)
                
            h_i=torch.stack(h_i,dim=0)
            h_i=self.encoder1(h_i).mean(dim=0)
            h.append(h_i)
        
        h=torch.stack(h,dim=0)
        h=self.encoder3(h);
        
        h=torch.tanh(h)*self.margin;
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];
    
