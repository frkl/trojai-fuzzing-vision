# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import logging
import os
import json
import jsonpickle
import pickle
import numpy as np

from sklearn.ensemble import RandomForestRegressor

import utils.models
from utils.abstract import AbstractDetector
from utils.models import load_model, load_models_dirpath

import torch
import torchvision
import skimage.io

import util.smartparse as smartparse
import util.db as db
import helper_r13_v3 as helper
import trinity_predv7 as trinity
import crossval

def center_to_corners_format(x):
    """
    Converts a PyTorch tensor of bounding boxes of center format (center_x, center_y, width, height) to corners format
    (x_0, y_0, x_1, y_1).
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - (0.5 * w)), (y_c - (0.5 * h)), (x_c + (0.5 * w)), (y_c + (0.5 * h))]
    return torch.stack(b, dim=-1)


class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))
        self.params=smartparse.dict2obj(metaparameters)
        
        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        
    

    def automatic_configure(self, models_dirpath: str):
        dataset=trinity.extract_dataset(models_dirpath,ts_engine=trinity.ts_engine,params=self.params)
        splits=crossval.split(dataset,self.params)
        ensemble,perf=crossval.train(splits,self.params)
        torch.save(ensemble,os.path.join(self.learned_parameters_dirpath,'model.pt'))
        return True

    def manual_configure(self, models_dirpath: str):
        return self.automatic_configure(models_dirpath)
    
    def infer(self,model_filepath,result_filepath,scratch_dirpath,examples_dirpath,round_training_dataset_dirpath):
        #Instantiate interface
        params=smartparse.obj()
        params.model_filepath=model_filepath;
        params.examples_dirpath=examples_dirpath;
        interface=helper.engine(params=params)
        
        #Extract features
        fvs=trinity.extract_fv(interface,trinity.ts_engine,self.params);
        fvs=db.Table.from_rows([fvs]).cuda();
        
        #Load model
        if not self.learned_parameters_dirpath is None:
            try:
                ensemble=torch.load(os.path.join(self.learned_parameters_dirpath,'model.pt'));
            except:
                ensemble=torch.load(os.path.join('/',self.learned_parameters_dirpath,'model.pt'));
            
            #print(ensemble)
            trojan_probability=trinity.predict(ensemble,fvs)
            
        else:
            trojan_probability=0.5;
        
        print(trojan_probability)
        
        with open(result_filepath, "w") as fp:
            fp.write('%f'%trojan_probability)
        
        logging.info("Trojan probability: %f", trojan_probability)
        return trojan_probability
    
    
