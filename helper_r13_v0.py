
import time
import copy
import os
import numpy as np
import torch
import json
import jsonschema
import jsonpickle
import warnings

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets.folder

import util.smartparse as smartparse

from pathlib import Path

warnings.filterwarnings("ignore")

def root():
    return '../trojai-datasets/object-detection-feb2023'

#The user provided engine that our algorithm interacts with
class engine:
    def __init__(self,folder=None,params=None):
        default_params=smartparse.obj();
        default_params.model_filepath='';
        default_params.examples_dirpath='';
        params=smartparse.merge(params,default_params)
        
        if params.model_filepath=='':
            params.model_filepath=os.path.join(folder,'model.pt');
        
        import utils.models
        from utils.abstract import AbstractDetector
        from utils.models import load_model, load_models_dirpath
        
        model, model_repr, model_class = load_model(params.model_filepath)
        self.model=model.cuda()
    