import torch
import torchvision.models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
import torchvision.transforms as Ts
import PIL.Image as Image
from torch.nn.utils.weight_norm import WeightNorm
import math
from collections import OrderedDict as OrderedDict
import copy

class new(nn.Module):
    def __init__(self,nlayers,nh,sz=1):
        super(new,self).__init__()
        self.layers=nn.ModuleList();
        if nlayers==1:
            self.layers.append(nn.Conv2d(5,3,2*sz-1,padding=sz-1));
        else:
            self.layers.append(nn.Conv2d(5,nh,2*sz-1,padding=sz-1));
            for i in range(nlayers-2):
                self.layers.append(nn.Conv2d(nh,nh,2*sz-1,padding=sz-1));
            
            self.layers.append(nn.Conv2d(nh,3,2*sz-1,padding=sz-1));
        
        self.ind=torch.Tensor([[0.0008878111839294434, -0.3853873610496521, 0.28623104095458984, -0.5597807168960571, -0.5853376388549805], [-0.46080464124679565, 0.1976543664932251, -0.21326637268066406, 0.5088878870010376, -0.20051753520965576], [0.23690444231033325, -0.2076154351234436, 0.47897177934646606, -0.4893004894256592, -0.5161682367324829], [-0.48983949422836304, 0.053960561752319336, 0.44288909435272217, -0.8992043733596802, 0.9562160968780518], [0.13138699531555176, -0.4481852054595947, 0.3407275676727295, -0.13302254676818848, -0.23877930641174316], [0.3486543893814087, -0.0388217568397522, -0.44439321756362915, 0.4443662166595459, 0.592059850692749], [0.1422862410545349, -0.3160783052444458, -0.008788883686065674, 0.7194095849990845, 0.49172258377075195], [0.4007243514060974, -0.4406278729438782, -0.31769704818725586, -0.5295590162277222, 0.8705337047576904], [-0.25632286071777344, -0.2599070072174072, -0.3104700446128845, -0.5638859272003174, -0.5374215841293335], [-0.2270910143852234, 0.45321953296661377, 0.019218385219573975, -0.23089957237243652, -0.3531841039657593], [0.11429613828659058, 0.18204951286315918, -0.31915533542633057, -0.8591408729553223, 0.46422624588012695], [0.07397031784057617, 0.062339067459106445, 0.16358786821365356, -0.6602640151977539, -0.5925731658935547], [-0.33109039068222046, -0.14857172966003418, -0.02702462673187256, -0.19072604179382324, -0.7594246864318848], [-0.49744582176208496, 0.12133240699768066, -0.08876359462738037, -0.8056139945983887, 0.6739015579223633], [-0.3208840489387512, 0.10442286729812622, 0.12884825468063354, 0.8171062469482422, -0.012134790420532227], [-0.3194931149482727, -0.4976598620414734, -0.41422128677368164, -0.08886098861694336, 0.1886221170425415]]);
        return;
    
    def compute_output(self,h):
        for i in range(len(self.layers)-1):
            h=self.layers[i](h);
            h=F.relu(h);
        
        h=self.layers[-1](h);
        return h
    
    def forward(self,I,aug=False):
        N,_,h,w=I.shape;
        if aug:
            im_=[];
            for i in range(N):
                im_i=I[i];
                
                
                brightness_factor=float(torch.Tensor(1).uniform_(0.8,1.2));
                contrast_factor=float(torch.Tensor(1).uniform_(0.8,1.2));
                gamma=float(torch.Tensor(1).uniform_(0.8,1.2));
                hue_factor=float(torch.Tensor(1).uniform_(-0.1,0.1));
                
                im_i=Ft.adjust_brightness(im_i,brightness_factor);
                im_i=Ft.adjust_contrast(im_i,contrast_factor);
                im_i=Ft.to_tensor(Ft.adjust_gamma(Ft.to_pil_image(im_i.cpu()),gamma)).cuda();
                im_i=Ft.to_tensor(Ft.adjust_hue(Ft.to_pil_image(im_i.cpu()),hue_factor)).cuda();
                
                im_.append(im_i);
            
            I=torch.stack(im_,dim=0);
        
        grid=F.affine_grid(torch.Tensor([[1,0,0],[0,1,0]]).view(1,2,3),[1,1,h,w]).view(1,h,w,2).permute(0,3,1,2);
        grid=grid.to(I.device).repeat(I.shape[0],1,1,1);
        
        h=torch.cat((I-0.5,grid),dim=1);
        out=self.compute_output(h)+I-0.5; #xyrgb => rgb
        out=F.sigmoid(out*5);
        return out
        
    def complexity(self):
        ind=self.ind.view(16,5,1,1).cuda();
        out=self.compute_output(ind).view(-1);
        diff=((out-ind[:,:3,:,:])**2).mean()
        return diff 
        #l2=0;
        #for param in self.parameters():
        #    l2=l2+(param**2).sum();
        #
        #return l2;
        
    
    def diff(self,other):
        fv=self.characterize();
        fv_other=other.characterize();
        return -((fv_other-fv)**2).mean()
    
    
    def characterize(self):
        ind=self.ind.view(16,5,1,1).cuda();
        return self.compute_output(ind).view(-1); #48-dim
    