
import os
import numpy as np
import torch
import json
import jsonschema
import warnings

import torchvision
from torchvision import transforms
import torchvision.datasets.folder

import cv2
import torchvision.datasets.folder
import torchvision.transforms.functional as Ft
import torchvision.transforms as Ts


import torch.nn.functional as F
import util.smartparse as smartparse
import util.db as db



warnings.filterwarnings("ignore")

def root():
    return '../trojai-datasets/round11-train-dataset'

def loadim(fname):
    augmentation_transforms = Ts.Compose([Ts.ConvertImageDtype(torch.float)])
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # convert the image to a tensor
    # should be uint8 type, the conversion to float is handled later
    image = torch.as_tensor(img)
    
    # move channels first
    image = image.permute((2, 0, 1))
    
    # convert to float (which normalizes the values)
    image = augmentation_transforms(image)
    
    batch_data = image.unsqueeze(dim=0).cuda();
    return batch_data;



class engine:
    def __init__(self,folder=None,params=None):
        default_params=smartparse.obj();
        default_params.model_filepath='';
        default_params.examples_dirpath='';
        params=smartparse.merge(params,default_params)
        #print(vars(params))
        if params.model_filepath=='':
            params.model_filepath=os.path.join(folder,'model.pt');
        if params.examples_dirpath=='':
            params.examples_dirpath=os.path.join(folder,'clean-example-data');
        
        model=torch.load(params.model_filepath);
        self.model=model.cuda()
        self.model.eval();
        
        self.examples_dirpath=params.examples_dirpath
    
    def load_examples(self,examples_dirpath=None):
        if examples_dirpath is None:
            examples_dirpath=self.examples_dirpath
        
        imnames=[fname for fname in os.listdir(examples_dirpath) if fname.endswith('jpg') or fname.endswith('png')];
        
        data=[];
        for imname in imnames:
            im=loadim(os.path.join(examples_dirpath,imname)).cuda()
            f=open(os.path.join(examples_dirpath,imname[:-4]+'.json'),'r');
            for line in f:
                label=int(line)
                break;
            
            data.append({'im':im,'label':label});
        
        return db.Table.from_rows(data);
    
    def eval(self,data):
        images=data['im'].cuda();
        if len(images.shape)==3:
            images=images.unsqueeze(dim=0);
        
        scores=self.model(images)
        loss=F.cross_entropy(scores,torch.LongTensor([data['label']]).cuda());
        return loss

