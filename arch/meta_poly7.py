import torch
import torchvision.models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.nn.utils.weight_norm import WeightNorm
import math
from collections import OrderedDict as OrderedDict
import copy

from torch.distributions import Normal
def gaussian_kernel_1d(sigma: float, num_sigmas: float = 2.) -> torch.Tensor:
    radius = math.ceil(num_sigmas * sigma)
    support = torch.arange(-radius, radius + 1, dtype=torch.float)
    kernel = torch.distributions.Normal(loc=0, scale=sigma).log_prob(support).exp_()
    # Ensure kernel weights sum to 1, so that image brightness is not altered
    return kernel.mul_(1 / kernel.sum())

def gaussian_filter_2d_slice(img: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel_1d = gaussian_kernel_1d(sigma)  # Create 1D Gaussian kernel
    padding = len(kernel_1d) // 2  # Ensure that image size does not change
    # Convolve along columns and rows
    img = F.conv2d(img, weight=kernel_1d.view(1, 1, -1, 1).to(img.device), padding=(padding, 0))
    img = F.conv2d(img, weight=kernel_1d.view(1, 1, 1, -1).to(img.device), padding=(0, padding))
    return img

def gaussian_filter_2d(img,sigma):
    return torch.cat([gaussian_filter_2d_slice(x,sigma) for x in img.split(1,dim=1)],dim=1)

class trigger(torch.Tensor):
    def apply(self,I):
        return apply(self,I)

#Produces triggered image given trigger definition and input image
#x 4+C channels
# channel 0: size, logdiff
# channel 1,2: offset, pre-tanh
# channel 3: mask, pre-sigmoid
# channel 4+: patch, pre-sigmoid
def apply(x,I):
    N,C,H,W=I.shape
    assert N==1
    N,_,h,w=x.shape
    assert N==1
    
    offset=torch.tanh(x[0,1:3,:,:].mean(dim=[-1,-2])) #2
    sz=x[0,0:1,:,:].mean(dim=[-1,-2]) #1
    mask=torch.sigmoid(x[:,3:4,:,:])
    #nmask=torch.sigmoid(-x[:,3:4,:,:])
    patch=torch.sigmoid(x[:,4:,:,:])
    
    scale=max(h,w)/min(H,W)
    sz=torch.exp(-sz+torch.Tensor(1).uniform_(-0.3,0.3).to(sz.device))#/scale
    offset=(offset)*sz
    t=torch.stack([sz,sz*0,offset[0:1],sz*0,sz,offset[1:2]],dim=0).view(1,2,3);
    
    grid=F.affine_grid(t,torch.Size((1,C,H,W)))
    mask=F.grid_sample(mask,grid)
    #nmask=F.grid_sample(nmask,grid)
    patch=F.grid_sample(patch,grid)
    
    out=I*(1-mask)+patch*mask
    out=out.clamp(min=0,max=1)
    return out


# A simple network that optimizes the trigger directly
class direct(nn.Module):
    def __init__(self,c=3,h=64,w=64):
        super().__init__()
        self.h=nn.Parameter(torch.Tensor(1,4+c,h,w).uniform_(-0.1,0.1));
    
    def forward(self):
        return trigger(self.h)


class gaussian(nn.Module):
    def __init__(self,c=3,h=64,w=64,sigma=2):
        super().__init__()
        self.sigma=sigma
        self.h=nn.Parameter(torch.Tensor(1,4+c,h,w).uniform_(-0.1,0.1));
    
    def forward(self):
        return trigger(gaussian_filter_2d(self.h,self.sigma))

#k odd
class conv(nn.Module):
    def __init__(self,c=3,h=64,w=64,nh=72,k=9):
        super().__init__()
        self.h=nn.Parameter(torch.Tensor(1,nh,h,w).uniform_(-0.1,0.1));
        self.decoder=nn.Sequential(
            #nn.Conv2d(nh,nh,k,padding=k//2),
            #nn.ReLU(),
            nn.Conv2d(nh,c+4,k,padding=k//2));
    
    def forward(self):
        return trigger(self.decoder(self.h))

#k odd
class conv_fixed(nn.Module):
    def __init__(self,c=3,h=64,w=64,nh=72,k=9):
        super().__init__()
        self.h=nn.Parameter(torch.Tensor(1,nh,h,w).uniform_(-0.1,0.1));
        self.decoder=[nn.Sequential(
            #nn.Conv2d(nh,nh,k,padding=k//2),
            #nn.ReLU(),
            nn.Conv2d(nh,c+4,k,padding=k//2)).cuda()];
    
    def forward(self):
        return trigger(self.decoder[0](self.h))

#k odd
#nh mul of 3
class conv_hie(nn.Module):
    def __init__(self,c=3,h=64,w=64,nh=72,k=9):
        super().__init__()
        self.h=nn.Parameter(torch.Tensor(1,nh//4,h,w).uniform_(-0.1,0.1));
        self.decoder=nn.Sequential(
            #nn.Conv2d(nh,nh,k,padding=k//2),
            #nn.ReLU(),
            nn.Conv2d(nh,c+4,k,padding=k//2));
    
    def forward(self):
        h=self.h
        _,_,H,W=h.shape 
        h1=F.adaptive_avg_pool2d(F.adaptive_avg_pool2d(h,(H//4,W//4)),(H,W))
        h2=F.adaptive_avg_pool2d(F.adaptive_avg_pool2d(h,(H//16,W//16)),(H,W)) 
        h3=F.adaptive_avg_pool2d(F.adaptive_avg_pool2d(h,(H//64,W//64)),(H,W)) 
        h=torch.cat((h,h1,h2,h3),dim=1)
        return trigger(self.decoder(h))



