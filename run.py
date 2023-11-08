
import os
import torch

import util.smartparse as smartparse
import util.db as db
import helper_r13_v0 as helper
import trinity as trinity

import util.smartparse as smartparse
default_params=smartparse.obj();
default_params.detector='detector.pt'
default_params.model='model2.pt'
params=smartparse.parse(default_params);

#Instantiate interface
interface=helper.engine(params=smartparse.obj({'model_filepath':params.model}))

#Extract features
fvs=trinity.extract_fv(interface,trinity.ts_engine,params);
fvs=db.Table.from_rows([fvs]).cuda();

#Load model
ensemble=torch.load(params.detector)
trojan_probability=trinity.predict(ensemble,fvs)

print('Trojan probability %.4f'%trojan_probability)
