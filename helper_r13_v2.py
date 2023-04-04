
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
import util.db as db

import skimage.io
import utils.models
from utils.abstract import AbstractDetector
from utils.models import load_model, load_models_dirpath


warnings.filterwarnings("ignore")

def prepare_boxes(anns, image_id):
    if len(anns) > 0:
        boxes = []
        class_ids = []
        for gt in anns:
            boxes.append([gt['bbox'][k] for k in ['x1','y1','x2','y2']])
            class_ids.append(gt['label'])
        
        class_ids = np.stack(class_ids)
        boxes = np.stack(boxes)
    else:
        class_ids = np.zeros((0))
        boxes = np.zeros((0, 4))
    
    degenerate_boxes = (boxes[:, 2:] - boxes[:, :2]) < 8
    degenerate_boxes = np.sum(degenerate_boxes, axis=1)
    if degenerate_boxes.any():
        boxes = boxes[degenerate_boxes == 0, :]
        class_ids = class_ids[degenerate_boxes == 0]
    
    target = {}
    target['boxes'] = torch.as_tensor(boxes)
    target['labels'] = torch.as_tensor(class_ids).type(torch.int64)
    target['image_id'] = torch.as_tensor(image_id)
    return target

def prepare_boxes_detr(anns, image_id,imw,imh):
    if len(anns) > 0:
        boxes = []
        class_ids = []
        for gt in anns:
            box=smartparse.obj(gt['bbox'])
            x1=box.x1/imw
            x2=box.x2/imw
            y1=box.y1/imh
            y2=box.y2/imh
            cx=(x1+x2)/2
            cy=(y1+y2)/2
            w=x2-x1
            h=y2-y1
            boxes.append([cx,cy,w,h])
            class_ids.append(gt['label'])
        
        class_ids = np.stack(class_ids)
        boxes = np.stack(boxes)
    else:
        class_ids = np.zeros((0))
        boxes = np.zeros((0, 4))
    
    #degenerate_boxes = (boxes[:, 2:] - boxes[:, :2]) < 8
    #degenerate_boxes = np.sum(degenerate_boxes, axis=1)
    #if degenerate_boxes.any():
    #    boxes = boxes[degenerate_boxes == 0, :]
    #    class_ids = class_ids[degenerate_boxes == 0]
    
    target = {}
    target['boxes'] = torch.as_tensor(boxes).float()
    target['class_labels'] = torch.as_tensor(class_ids).type(torch.int64)
    target['image_id'] = torch.as_tensor(image_id)
    return target

def decode_boxes_detr(box,imw,imh):
    cx,cy,w,h=box[:,0],box[:,1],box[:,2],box[:,3]
    x1=cx-w/2
    y1=cy-h/2
    x2=x1+w
    y2=y1+h
    
    x1=x1*imw
    x2=x2*imw
    y1=y1*imh
    y2=y2*imh
    return torch.stack([x1,y1,x2,y2],dim=-1)

def visualize(imname,data=None,out='tmp.png',threshold=0.1):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots()
    
    if isinstance(imname,str):
        img = mpimg.imread(imname)
    elif torch.is_tensor(imname):
        img=imname.permute(1,2,0).cpu().numpy();
    else:
        img=transforms.ToTensor()(imname);
        img=img.permute(1,2,0).cpu().numpy();
    
    
    imgplot = ax.imshow(img) #HWC
    
    if not data is None:
        for i in range(len(data['boxes'])):
            bbox=data['boxes'][i]
            #label='L%d'%int(data['labels'][i])#id2label[int(data['labels'][i])];
            label=int(data['labels'][i]);
            score=float(data['scores'][i]);
            
            if score>threshold:
                x0,y0,x1,y1=int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
                rect=patches.Rectangle((x0,y0),x1-x0,y1-y0, linewidth=3, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                
                t=ax.text(x0,y0,'%d: %.3f'%(label,score),color='white',weight='bold',ha='left',va='top');
                t.set_bbox(dict(facecolor='black', alpha=0.5, edgecolor='black'))
    
    
    plt.savefig(out)
    plt.close();


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
        if params.examples_dirpath=='':
            params.examples_dirpath=os.path.join(folder,'clean-example-data');
        
        
        model, model_repr, model_class = load_model(params.model_filepath)
        self.model=model.cuda()
        self.model.train()
        #Turn off batch norm
        #But sets to training mode for loss & gradients
        for module in self.model.modules():
            module.eval()
            #if isinstance(module, nn.BatchNorm2d):
            #    module.eval()
        
        for param in self.model.parameters():
            param.requires_grad_(True)
        
        self.model_filepath=params.model_filepath
        self.examples_dirpath=params.examples_dirpath
    
    #Load data into memory, for faster processing.
    def load_examples(self,examples_dirpath=None):
        if examples_dirpath is None:
            examples_dirpath=self.examples_dirpath
        
        augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])
        fns=[os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.png')]
        fns.sort()
        
        images=[];
        image_paths=[];
        targets=[];
        anns=[];
        
        data=[];
        for fn in fns:
            fn_gt = fn.replace('.png','.json')
            with open(fn_gt, mode='r', encoding='utf-8') as f:
                gt = jsonpickle.decode(f.read())
            
            # load the example image
            img = skimage.io.imread(fn)
            image = torch.as_tensor(img)
            image = image.permute((2, 0, 1))
            image = augmentation_transforms(image).unsqueeze(0)
            target=prepare_boxes(gt,0)
            
            data.append({'image':image,'fname':fn,'gt':gt,'target':target})
        
        data=db.Table.from_rows(data)
        return data
    
    def eval_grad(self, data):
        im=data['image']
        if 'DetrForObjectDetection' in self.model.__class__.__name__:
            self.model.zero_grad()
            loss = self.model(im.cuda(),labels=[{k: v.cuda() for k, v in prepare_boxes_detr(data['gt'],0,im.shape[-1],im.shape[-2]).items()}])
            #print(loss['loss_dict'])
            loss=loss['loss']
            loss.backward()
            loss=float(loss)
            
            g=[]
            for w in self.model.parameters():
                if w.grad is None:
                    g.append((w*0).data.clone().cpu())
                else:
                    g.append(w.grad.data.clone().cpu())
        else:
            m=copy.deepcopy(self.model)
            m=m.train()
            m.zero_grad()
            loss = m(im.cuda(),[{k: v.cuda() for k, v in prepare_boxes(data['gt'],0).items()}])
            loss=torch.stack([loss[k] for k in loss],dim=0).sum()
            loss.backward()
            loss=float(loss)
            
            g=[]
            for w in m.parameters():
                if w.grad is None:
                    g.append(None)
                else:
                    g.append(w.grad.data.clone().cpu())
        
        return g
    
    def eval_loss(self, data):
        im=data['image']
        if 'DetrForObjectDetection' in self.model.__class__.__name__:
            if torch.is_tensor(data['gt']):
                print(data['gt'].shape)
                print(data['image'].shape)
            
            loss = self.model(im.cuda(),labels=[{k: v.cuda() for k, v in prepare_boxes_detr(data['gt'],0,im.shape[-1],im.shape[-2]).items()}])
            loss=loss['loss']
            
        else:
            loss = self.model(im.cuda(),[{k: v.cuda() for k, v in prepare_boxes(data['gt'],0).items()}])
            loss=torch.stack([loss[k] for k in loss],dim=0).sum()
        
        return loss
    
    def eval(self, data):
        im=data['image']
        outputs=self.model(im.cuda())
        if 'DetrObjectDetectionOutput' in outputs.__class__.__name__:
            boxes = decode_boxes_detr(outputs.pred_boxes[0],im.shape[-1],im.shape[-2])
            print(boxes.shape)
            logits = outputs.logits[0]
            probs = F.softmax(logits,dim=-1)[:,:-1]
            scores,labels=probs.max(dim=-1)
        else:
            outputs=outputs[0]
            boxes = outputs['boxes']
            scores = outputs['scores']
            labels= outputs['labels']
        
        #visualize(data['fname'],{'scores':scores,'labels':labels,'boxes':boxes},threshold=0.1)
        return boxes,scores
    
    def eval_hidden(self,data):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.data.clone()
            return hook
        
        nlayers=len(list(self.model.children()))
        for i,layer in enumerate(self.model.children()):
            layer.register_forward_hook(get_activation(i));
        
        scores=self.model(data['fv'].cuda())
        
        h=[activation[i] for i in range(nlayers)];
        return h
