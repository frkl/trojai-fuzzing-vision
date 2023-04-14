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
#import segmentation_models_pytorch as smp

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

import copy
class trigger:
    def __init__(self,net):
        self.net=net
    
    def __call__(self,*argc,**argv):
        return self.net(*argc,**argv)
    
    def detach(self):
        return trigger(copy.deepcopy(self.net));
    
    def clone(self):
        return trigger(copy.deepcopy(self.net));
        
    def apply(self,I):
        return self.net.apply(I)

#Produces triggered image given trigger definition and input image
#x 4+C channels
# channel 0: size, logdiff
# channel 1,2: offset, pre-tanh
# channel 3: mask, pre-sigmoid
# channel 4+: patch, pre-sigmoid
def apply(trigger,I):
    out=trigger.net.apply(I)
    out=out.clamp(min=0,max=1)
    return out


# A simple network that optimizes the trigger directly
class direct(nn.Module):
    def __init__(self,nlayers=2,nh=16,sz=1):
        super().__init__()
        self.layers=nn.ModuleList();
        if nlayers==1:
            self.layers.append(nn.Conv2d(5,3,2*sz-1,padding=sz-1));
        else:
            self.layers.append(nn.Conv2d(5,nh,2*sz-1,padding=sz-1));
            for i in range(nlayers-2):
                self.layers.append(nn.Conv2d(nh,nh,2*sz-1,padding=sz-1));
            
            self.layers.append(nn.Conv2d(nh,3,2*sz-1,padding=sz-1));
    
    
    def compute_output(self,h):
        for i in range(len(self.layers)-1):
            h=self.layers[i](h);
            h=F.relu(h);
        
        h=self.layers[-1](h);
        return h
    
    def apply(self,I):
        N,_,h,w=I.shape;
        
        grid=F.affine_grid(torch.Tensor([[1,0,0],[0,1,0]]).view(1,2,3),[1,1,h,w]).view(1,h,w,2).permute(0,3,1,2);
        grid=grid.to(I.device).repeat(I.shape[0],1,1,1);
        
        h=torch.cat((I-0.5,grid),dim=1);
        out=self.compute_output(h)+I-0.5; #xyrgb => rgb
        out=F.sigmoid(out*5);
        return out
    
    def forward(self):
        return trigger(self)
    
    def detach(self):
        return copy.deepcopy(self)

