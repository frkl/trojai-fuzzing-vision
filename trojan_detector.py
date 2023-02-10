# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import cv2
import torch
import torchvision
import json
import jsonschema
import math
import logging
import warnings 
import util.db as db
warnings.filterwarnings("ignore")

import torch.nn.functional as F
import util.smartparse as smartparse
import util.db as db
import copy
import trinity as trinity
import crossval
import helper_r11_v2 as helper

def example_trojan_detector(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, round_training_dataset_dirpath, parameters_dirpath, params):
    
    p=smartparse.obj()
    p.model_filepath=model_filepath;
    p.examples_dirpath=examples_dirpath;
    interface=helper.engine(params=p)
    
    #Extract features
    fvs=trinity.extract_fv(interface,trinity.ts_engine,params);
    fvs=db.Table.from_rows([fvs]).cuda();
    
    #Load model
    if not parameters_dirpath is None:
        try:
            ensemble=torch.load(os.path.join(parameters_dirpath,'model.pt'));
        except:
            ensemble=torch.load(os.path.join('/',parameters_dirpath,'model.pt'));
        
        #print(ensemble)
        trojan_probability=trinity.predict(ensemble,fvs)
        
    else:
        trojan_probability=0.5;
    
    print(trojan_probability)
    
    while True:
        try:
            with open(result_filepath, 'w') as fh:
                fh.write("{}".format(trojan_probability))
            
            break;
        except:
            pass;
    
    logging.info("Trojan probability: %f", trojan_probability)
    return;


def configure(learned_parameters_dirpath,
              models_dirpath,
              params):
    dataset=trinity.extract_dataset(models_dirpath,ts_engine=trinity.ts_engine,params=params)
    splits=crossval.split(dataset,params)
    ensemble=crossval.train(splits,params)
    torch.save(ensemble,os.path.join(learned_parameters_dirpath,'model.pt'))


if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='')
    parser.add_argument('--round_training_dataset_dirpath', type=str, help='File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.', default=None)

    parser.add_argument('--metaparameters_filepath', help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.', action=ActionConfigFile)
    parser.add_argument('--schema_filepath', type=str, help='Path to a schema file in JSON Schema format against which to validate the config file.', default=None)
    parser.add_argument('--learned_parameters_dirpath', type=str, help='Path to a directory containing parameter data (model weights, etc.) to be used when evaluating models.  If --configure_mode is set, these will instead be overwritten with the newly-configured parameters.')

    parser.add_argument('--configure_mode', help='Instead of detecting Trojans, set values of tunable parameters and write them to a given location.', default=False, action="store_true")
    parser.add_argument('--configure_models_dirpath', type=str, help='Path to a directory containing models to use when in configure mode.')

    # these parameters need to be defined here, but their values will be loaded from the json file instead of the command line
    parser.add_argument('--szcap', type=int, help='An example tunable parameter.')
    parser.add_argument('--example_img_format', type=str, default='png')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")
    logging.info("example_trojan_detector.py launched")

    # Validate config file against schema
    config_json = None
    if args.metaparameters_filepath is not None:
        with open(args.metaparameters_filepath[0]()) as config_file:
            config_json = json.load(config_file)
    
    if args.schema_filepath is not None:
        with open(args.schema_filepath) as schema_file:
            schema_json = json.load(schema_file)

        # this throws a fairly descriptive error if validation fails
        jsonschema.validate(instance=config_json, schema=schema_json)
    
    logging.info(args)
    
    if not args.configure_mode:
        if (args.model_filepath is not None and
            args.result_filepath is not None and
            args.scratch_dirpath is not None and
            args.examples_dirpath is not None and
            args.round_training_dataset_dirpath is not None and
            args.learned_parameters_dirpath is not None):

            logging.info("Calling the trojan detector")
            example_trojan_detector(args.model_filepath,
                                    args.result_filepath,
                                    args.scratch_dirpath,
                                    args.examples_dirpath,
                                    args.round_training_dataset_dirpath,
                                    args.learned_parameters_dirpath,params=args)
        else:
            logging.info("Required Evaluation-Mode parameters missing!")
    else:
        if (args.learned_parameters_dirpath is not None and
            args.configure_models_dirpath is not None):

            logging.info("Calling configuration mode")
            # all 3 example parameters will be loaded here, but we only use parameter3
            configure(args.learned_parameters_dirpath,
                      args.configure_models_dirpath,
                      params=args)
        else:
            logging.info("Required Configure-Mode parameters missing!")


