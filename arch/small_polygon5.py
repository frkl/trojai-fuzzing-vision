import torch
import torchvision.models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import math
from collections import OrderedDict as OrderedDict
import copy

def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val



class masked_polygon:
    def __init__(self,v,offset,sz):
        self.v=F.pad(v,(5,5,5,5),value=0)
        self.offset=offset
        self.sz=sz.view(-1)
    
    def __call__(self,I):
        N,C,H,W=I.shape
        _,h,w=self.v.shape
        
        trigger=self.v[:-1,:,:].unsqueeze(0);
        mask=self.v[-1:,:,:].unsqueeze(0);
        
        transform=[];
        for j in range(N):
            scale=max(h,w)/min(H,W)
            sz=torch.exp(-self.sz)/scale; 
            #sz=sz*0+2; 
            offset=self.offset[j]*sz
            t=torch.stack([sz,sz*0,offset[0:1],sz*0,sz,offset[1:2]],dim=0).view(2,3);
            transform.append(t);
        
        transform=torch.stack(transform,dim=0)
        grid=F.affine_grid(transform,I.size()).to(mask.device);
        #Synthesize trigger
        mask=grid_sample(mask.repeat(N,1,1,1),grid);
        trigger=grid_sample(trigger.repeat(N,1,1,1),grid);
        out=I*(1-mask)+trigger*mask;
        
        return out;
    
    def size(self):
        return self.sz



class new(nn.Module):
    def __init__(self,nim):
        super(new,self).__init__()
        nh=12
        self.net=nn.Sequential(nn.Conv2d(nh*3,4+1+2*nim,9,padding=4),nn.ReLU(),nn.Conv2d(4+1+2*nim,4+1+2*nim,9,padding=4))
        self.nim=nim
        return;
    
    def forward(self,x):
        _,H,W=x.shape
        h0=x.unsqueeze(0)
        h1=F.adaptive_avg_pool2d(F.adaptive_avg_pool2d(x.unsqueeze(0),(H//4,W//4)),(H,W))
        h2=F.adaptive_avg_pool2d(F.adaptive_avg_pool2d(x.unsqueeze(0),(H//16,W//16)),(H,W)) 
        h=torch.cat((h0,h1,h2),dim=1);
        h=self.net(h);
        
        v=torch.sigmoid(h[0,:4,:,:])
        offset=torch.tanh(h[0,-2*self.nim:,:,:].mean(dim=-1).mean(dim=-1).view(self.nim,2))
        sz=h[0,4,:,:].mean().view(-1);
        return masked_polygon(v,offset,sz);

    
