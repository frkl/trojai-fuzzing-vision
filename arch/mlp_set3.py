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

#Input n x K x K x ninput
#Output n x K x 2nh
class same_diff_encoder(nn.Module):
    def __init__(self,ninput,nh,nlayers):
        super().__init__()
        self.encoder=MLP(ninput,nh,nh,nlayers);
    
    def forward(self,fv):
        rounds=fv.shape[0];
        nclasses=fv.shape[1];
        assert fv.shape[2]==nclasses;
        
        h=fv.view(rounds*nclasses*nclasses,-1);
        h=self.encoder(h);
        h=h.view(rounds,nclasses*nclasses,-1);
        
        ind_diag=list(range(0,nclasses*nclasses,nclasses+1));
        ind_off_diag=list(set(list(range(nclasses*nclasses))).difference(set(ind_diag)))
        ind_diag=torch.LongTensor(list(ind_diag)).to(h.device)
        ind_off_diag=torch.LongTensor(list(ind_off_diag)).to(h.device)
        h_diag=h[:,ind_diag,:];
        h_off_diag=h[:,ind_off_diag,:].contiguous().view(rounds,nclasses,nclasses-1,-1).mean(2);
        return torch.cat((h_diag,h_off_diag),dim=2);

#Input n x K x ninput
#Output n x nh
class encoder(nn.Module):
    def __init__(self,ninput,nh,nlayers):
        super().__init__()
        self.encoder=MLP(ninput,nh,nh,nlayers);
    
    def forward(self,fv):
        rounds=fv.shape[0];
        nclasses=fv.shape[1];
        
        h=fv.view(rounds*nclasses,-1);
        h=self.encoder(h);
        h=h.view(rounds,nclasses,-1).mean(1);
        return h;


class new(nn.Module):
    def __init__(self,params):
        super(new,self).__init__()
        
        
        nh=params.nh;
        nh2=params.nh2;
        nh3=params.nh3;
        nlayers=params.nlayers
        nlayers2=params.nlayers2
        nlayers3=params.nlayers3
        
        self.encoder1=MLP(30,nh,nh,nlayers);
        self.encoder2=MLP(3*nh,nh2,nh2,nlayers2);
        self.encoder_combined=MLP(nh2,nh3,2,nlayers3);
        
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        
        return;
    
    def forward(self,data_batch):
        h=[];
        #Have to process one by one due to variable nim & nclasses
        for i in range(len(data_batch['fvs_poly'])):
            #Encode the color features
            fvs_polygon=data_batch['fvs_poly'][i].cuda()
            ntriggers,npts,nexamples,nclasses=fvs_polygon.shape
            
            h_i=fvs_polygon.permute(0,2,3,1).view(ntriggers,nexamples*nclasses,npts).contiguous()
            h_i=self.encoder1(h_i).view(ntriggers,nexamples*nclasses,-1)
            h_i=torch.cat((h_i.mean(dim=1),h_i.max(dim=1)[0],h_i.min(dim=1)[0]),dim=1);
            h_i=self.encoder2(h_i).mean(dim=0);
            h.append(h_i);
        
        h=torch.stack(h,dim=0);
        #h=torch.cat((h,attrib_logits,attrib_histogram,filter_log,n),dim=1); 
        #print(h.shape);
        h=self.encoder_combined(h);
        h=torch.tanh(h)*8;
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];
    