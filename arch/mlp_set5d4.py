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

#XXh=>8h
class inv_ii_2(nn.Module):
    def forward(self,x):
        x=F.pad(x,(0,1),value=1)
        
        #1st order
        einstr_1=['aaZ->Z']
        einstr_1+=['abZ->Z']
        
        #2nd order
        einstr_2=['aaZ,aaZ->Z']
        einstr_2+=['aaZ,abZ->Z']
        einstr_2+=['abZ,abZ->Z']
        einstr_2+=['baZ,abZ->Z']
        einstr_2+=['abZ,acZ->Z']
        einstr_2+=['abZ,caZ->Z']
        
        h=[]
        for s in einstr_1:
            h.append(torch.einsum(s,x))
        
        for s in einstr_2:
            h.append(torch.einsum(s,x,x))
        
        h=[s[:-1]/(s[-1:].abs()+1e-20) for s in h]
        h=torch.stack(h,dim=0) # 8xn
        return h

class vector_log(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()*torch.log1p(x.abs())/10
    
    @staticmethod
    def backward(ctx, grad_output):
        x,=ctx.saved_tensors
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
        
        self.pool=inv_ii_2()
        self.encoder1=MLP(27*3*4,nh,nh2,nlayers)
        self.encoder2=MLP(8*nh2,nh3,2,nlayers3);
        
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        
        return;
    
    def forward(self,data_batch):
        with torch.no_grad():
            fvs_=[]
            for i in range(len(data_batch['fvs_fg_clean'])):
                fv=torch.cat([data_batch[s][i][:27].to(self.w.device) for s in ['fvs_fg_clean','fvs_fg_triggered','fvs_bg_clean','fvs_bg_triggered']],dim=-1)
                nt,nc,_,nh=fv.shape
                fv=fv.permute(1,2,0,3).contiguous().view(nc,nc,nt*nh)
                fvs_.append(fv)
        
        fvs=[self.encoder1(fv) for fv in fvs_];
        fvs=[self.pool(fv).view(-1) for fv in fvs];
        h_=torch.stack(fvs,dim=0);
        
        h=vector_log(h_)
        
        h=self.encoder2(h);
        h=torch.tanh(h)*self.margin;
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];
    
