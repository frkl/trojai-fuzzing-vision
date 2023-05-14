
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

from pathlib import Path

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
    target['labels'] = torch.as_tensor(class_ids).type(torch.int64)+1
    target['image_id'] = torch.as_tensor(image_id)
    return target

def prepare_boxes_as_prediction(anns):
    if len(anns) > 0:
        boxes=[]
        labels=[]
        scores=[]
        for gt in anns:
            boxes.append([gt['bbox'][k] for k in ['x1','y1','x2','y2']])
            labels.append(gt['label']+1)
            scores.append(1.0)
        
        boxes=torch.Tensor(boxes)
        labels=torch.LongTensor(labels)
        scores=torch.Tensor(scores)
    else:
        boxes = torch.zeros((0,4))
        labels = torch.zeros((0)).long()
        scores = torch.zeros((0))
    
    return {'boxes':boxes,'labels':labels,'scores':scores}

    

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
    target['class_labels'] = torch.as_tensor(class_ids).type(torch.int64)+1
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
    
    
    if out.rfind('/')>=0:
        Path(out[:out.rfind('/')]).mkdir(parents=True, exist_ok=True);
    
    plt.savefig(out)
    plt.close();

def iou(box1,box2):
    x00,y00,x01,y01=box1
    x10,y10,x11,y11=box2
    x00,x01=min(x00,x01),max(x00,x01)
    y00,y01=min(y00,y01),max(y00,y01)
    x10,x11=min(x10,x11),max(x10,x11)
    y10,y11=min(y10,y11),max(y10,y11)
    
    overlap_x=max(min(x01,x11)-max(x00,x10),0)
    overlap_y=max(min(y01,y11)-max(y00,y10),0)
    
    i=overlap_x*overlap_y
    u=max((x01-x00)*(y01-y00)+(x11-x10)*(y11-y10)-i,1e-20)
    
    return i/u

def compare_boxes(pred0,pred1,nclasses):
    s={};
    bbox0=pred0['boxes'].tolist()
    bbox1=pred1['boxes'].tolist()
    for i in range(len(pred0['labels'])):
        for j in range(len(pred1['labels'])):
            v=iou(bbox0[i],bbox1[j])
            if v>0:
                c0=int(pred0['labels'][i])
                c1=int(pred1['labels'][j])
                s0=float(pred0['scores'][i])
                s1=float(pred1['scores'][j])
                if not (c0,c1) in s:
                    s[(c0,c1)]=[]
                
                s[(c0,c1)].append([v,s0,s1])
    
    #Generate confusion matrix
    scores=torch.zeros(nclasses+1,nclasses+1)
    for c0,c1 in s:
        v=0
        for box in s[(c0,c1)]:
            if box[0]>0.5:
                v+=box[1]*box[2]
        
        scores[(c0,c1)]=v
    
    
    '''
    indicies=[]
    values=[]
    #Generate an invariant encoding
    for k in s:
        v=torch.Tensor(s[k])
        n=torch.Tensor([len(v)])
        m=v.mean(dim=0)
        v=v-m
        cov=torch.mm(v.t(),v)/n
        h=torch.cat([n,m,cov.view(-1)],dim=0)
        indicies.append(k)
        values.append(h)
    
    indicies=torch.Tensor(indicies)
    values=torch.stack(values,dim=0)
    h=torch.sparse_coo_tensor(indicies.t(),values,(nclasses,nclasses,13));
    '''
    return scores

#h=compare_boxes(pred,pred,64)


def root():
    return '../trojai-datasets/object-detection-feb2023'

def get(id):
    folder=os.path.join(root(),'models','id-%08d'%id)
    return engine(folder)

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
        self.enable_loss()
        
        self.root=params.model_filepath[:params.model_filepath.rfind('/')]
        self.model_filepath=params.model_filepath
        self.examples_dirpath=params.examples_dirpath
    
    def nclasses(self):
        #Load config file
        f=open(os.path.join(self.root,'config.json'),'r')
        config=json.load(f)
        f.close()
        return config['py/state']['number_classes']
    
    def trojaned(self):
        fname=os.path.join(self.root,'trigger_0.png')
        if os.path.exists(fname):
            return True
        else:
            return False
    
    
    def load_patch(self,r=0,g=0,b=0):
        return torch.Tensor([r,g,b,1]).view(4,1,1)
    
    
    def paste_trigger(self,image,trigger,bbox):
        x1,y1,x2,y2=bbox
        patch=F.adaptive_avg_pool2d(trigger,(y2-y1,x2-x1)).to(image.device)
        patch=patch[0:3]*patch[3:]+image[:,y1:y2,x1:x2]*(1-patch[3:])
        new_image=image.clone()
        new_image[:,y1:y2,x1:x2]=patch
        return new_image
    
    def get_triggers(self):
        augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])
        #Load Trojan definition
        f=open(os.path.join(self.root,'config.json'),'r')
        config=json.load(f)
        f.close()
        
        triggers=[]
        for i,trigger in enumerate(config['py/state']['triggers']):
            fname=os.path.join(self.root,'trigger_%d.png'%i)
            img = skimage.io.imread(fname)
            img = torch.as_tensor(img)
            img = img.permute((2, 0, 1))
            img = augmentation_transforms(img)
            
            source=trigger['py/state']['source_class']
            if trigger['py/object']=='object_detection_trigger_executor.ObjectDetectionMisclassificationPolygonTriggerExecutor':
                #Cause misclassifications when pasted on sign
                behavior='misclassify'
            elif trigger['py/object']=='object_detection_trigger_executor.ObjectDetectionEvasionPolygonTriggerExecutor':
                #Remove bounding box when pasted on sign
                behavior='evasion'
            elif trigger['py/object']=='object_detection_trigger_executor.ObjectDetectionInjectionPolygonTriggerExecutor':
                #Induces new bounding boxes of source/target when pasted in image
                behavior='injection'
            elif trigger['py/object']=='object_detection_trigger_executor.ObjectDetectionLocalizationPolygonTriggerExecutor':
                #Move bounding boxes of source/target when pasted in image
                behavior='localization'
            else:
                print(trigger['py/object'])
                a=0/0
            
            triggers.append({'source':source,'behavior':behavior,'trigger':img})
        
        return triggers
    
    def trojan_examples(self,data,trigger,source=None,suffix=''):
        data=copy.deepcopy(data)
        
        for i in range(len(data)):
            new_image=data[i]['image'].clone().squeeze(0);
            _,H,W=new_image.shape
            
            if source is None:
                #Paste at random location/size
                sz=int(torch.LongTensor(1).random_(10,30))
                x=int(torch.LongTensor(1).random_(0,W-sz+1))
                y=int(torch.LongTensor(1).random_(0,H-sz+1))
                new_image=self.paste_trigger(new_image,trigger,(x,y,x+sz,y+sz))
            elif isinstance(source,str) and source=='any':
                #Paste on any object
                found=0
                if len(data[i]['gt'])>0:
                    ind=int(torch.LongTensor(1).random_(len(data[i]['gt'])))
                    bbox=data[i]['gt'][ind]
                    x1,y1,x2,y2=[int(bbox['bbox'][s]) for s in ['x1','y1','x2','y2']]
                    #Shrink area to 70% so we get to center of the object
                    cx=(x1+x2)/2
                    cy=(y1+y2)/2
                    w=x2-x1
                    h=y2-y1
                    
                    x1=int(cx-w*0.35)
                    x2=int(cx+w*0.35)
                    y1=int(cy-h*0.35)
                    y2=int(cy+h*0.35)
                    
                    sz=int(torch.LongTensor(1).random_(15,40))
                    sz=min(min(sz,x2-x1),y2-y1)
                    x=int(torch.LongTensor(1).random_(x1,x2-sz+1))
                    y=int(torch.LongTensor(1).random_(y1,y2-sz+1))
                    new_image=self.paste_trigger(new_image,trigger,(x,y,x+sz,y+sz))
                else:
                    #No source object. Paste at random location
                    sz=int(torch.LongTensor(1).random_(10,30))
                    x=int(torch.LongTensor(1).random_(0,W-sz+1))
                    y=int(torch.LongTensor(1).random_(0,H-sz+1))
                    new_image=self.paste_trigger(new_image,trigger,(x,y,x+sz,y+sz))
                
            else:
                #Paste on source object
                found=0
                for bbox in data[i]['gt']:
                    if bbox['label']==source:
                        x1,y1,x2,y2=[int(bbox['bbox'][s]) for s in ['x1','y1','x2','y2']]
                        #Shrink area to 70% so we get to center of the object
                        cx=(x1+x2)/2
                        cy=(y1+y2)/2
                        w=x2-x1
                        h=y2-y1
                        
                        x1=int(cx-w*0.35)
                        x2=int(cx+w*0.35)
                        y1=int(cy-h*0.35)
                        y2=int(cy+h*0.35)
                        
                        sz=int(torch.LongTensor(1).random_(15,40))
                        sz=min(min(sz,x2-x1),y2-y1)
                        x=int(torch.LongTensor(1).random_(x1,x2-sz+1))
                        y=int(torch.LongTensor(1).random_(y1,y2-sz+1))
                        new_image=self.paste_trigger(new_image,trigger,(x,y,x+sz,y+sz))
                        found=1
                        break
                
                if found==0:
                    #No source object. Paste at random location
                    sz=int(torch.LongTensor(1).random_(10,30))
                    x=int(torch.LongTensor(1).random_(0,W-sz+1))
                    y=int(torch.LongTensor(1).random_(0,H-sz+1))
                    new_image=self.paste_trigger(new_image,trigger,(x,y,x+sz,y+sz))
            
            data[i]['image']=new_image.unsqueeze(0)
            data[i]['fname']+=suffix
        
        return data
    
    def add_object(self,data,class_id=-1,suffix=''):
        '''
        f=open(os.path.join(self.root,'config.json'),'r')
        config=json.load(f)
        f.close()
        class_id=[trigger['py/state']['target_class'] for trigger in config['py/state']['triggers']][0]
        '''
        
        #Load sign
        fg_fname=os.path.join(self.root,'fg_class_translation.json')
        
        if not os.path.exists(fg_fname):
            return []
        else:
            f=open(os.path.join(self.root,'fg_class_translation.json'),'r')
            class2fname=json.load(f)
            f.close()
        
        fname=class2fname[str(class_id)]
        fname=os.path.join(self.root,'foregrounds',fname)
        img = skimage.io.imread(fname)
        img = torch.as_tensor(img)
        img = img.permute((2, 0, 1))
        transform = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])
        img=transform(img);
        
        #Paste at random locations, 70px to 30px
        data=copy.deepcopy(data);
        for i in range(len(data)):
            sz=int(torch.LongTensor(1).random_(30,100));
            x=int(torch.LongTensor(1).random_(256-sz));
            y=int(torch.LongTensor(1).random_(256-sz));
            
            data[i]['gt'].append({'bbox':{'x1':x,'y1':y,'x2':x+sz,'y2':y+sz},'label':class_id})
            patch=F.adaptive_avg_pool2d(img,(sz,sz))
            data[i]['image'][0,:,y:y+sz,x:x+sz]=data[i]['image'][0,:,y:y+sz,x:x+sz]*(1-patch[3:])+patch[:3]*patch[3:];
            data[i]['fname']+=suffix
        
        return data
    
    
    
    
    def enable_loss(self):
        self.model.train()
        #Turn off batch norm
        #But sets to training mode for loss & gradients
        for module in list(self.model.modules())[1:]:
            #module.eval()
            if not any([s in module.__class__.__name__ for s in ('RCNN')]): #,'RPN','RoI'
                module.eval()
        
        for param in self.model.parameters():
            param.requires_grad_(True)
        
        mc=list(set([module.__class__.__name__ for module in list(self.model.modules())[1:]]))
        #print(mc)
        return
    
    
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
            
            data.append({'image':image,'fname':fn,'gt':gt})
        
        #data=db.Table.from_rows(data)
        return data
    
    def cleanse_gt(self,data):
        data=copy.deepcopy(data)
        #Load sign
        f=open(os.path.join(self.root,'config.json'),'r')
        config=json.load(f)
        f.close()
        
        label_map={}
        label_delete=[]
        
        for trigger in config['py/state']['triggers']:
            if trigger['py/object']=='object_detection_trigger_executor.ObjectDetectionMisclassificationPolygonTriggerExecutor':
                source=trigger['py/state']['source_class']
                target=trigger['py/state']['target_class']
                label_map[target]=source
            elif trigger['py/object']=='object_detection_trigger_executor.ObjectDetectionInjectionPolygonTriggerExecutor':
                source=trigger['py/state']['source_class']
                label_delete.append(source)
            else:
                print(trigger['py/object'])
                a=0/0
        
        for i in range(len(data)):
            new_gt=[];
            for bbox in data[i]['gt']:
                if bbox['label'] in label_map:
                    bbox['label']=label_map[bbox['label']]
                    new_gt.append(bbox)
                elif bbox['label'] in label_delete:
                    pass;
                else:
                    new_gt.append(bbox)
            
            data[i]['gt']=new_gt
        
        return data
        
    def more_clean_examples(self,class_id=-1,v=0.4,suffix=''):
        '''
        f=open(os.path.join(self.root,'config.json'),'r')
        config=json.load(f)
        f.close()
        class_id=[trigger['py/state']['target_class'] for trigger in config['py/state']['triggers']][0]
        '''
        
        #Load sign
        fg_fname=os.path.join(self.root,'fg_class_translation.json')
        
        if not os.path.exists(fg_fname):
            return []
        else:
            f=open(os.path.join(self.root,'fg_class_translation.json'),'r')
            class2fname=json.load(f)
            f.close()
        
        augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])
        
        #Paste at random locations, 70px to 30px
        data=[];
        if class_id<=0:
            classes=[int(i) for i in class2fname.keys()];
        else:
            classes=[class_id]
        
        for cid in classes:
            fname=class2fname[str(cid)];
            fname=os.path.join('foregrounds',fname)
            transform=transforms.Compose([lambda image: image.convert("RGB"),transforms.ToTensor()]);
            img=torchvision.datasets.folder.default_loader(os.path.join(self.root,fname));
            img=transform(img);
            img = augmentation_transforms(img).unsqueeze(0)
            
            im=torch.Tensor(1,3,256,256).fill_(v)
            sz=int(torch.LongTensor(1).random_(30,100));
            x=int(torch.LongTensor(1).random_(256-sz));
            y=int(torch.LongTensor(1).random_(256-sz));
            patch=F.adaptive_avg_pool2d(img,(sz,sz))
            im[:,:,y:y+sz,x:x+sz]=patch;
            
            gt=[{'bbox':{'x1':x,'y1':y,'x2':x+sz,'y2':y+sz},'label':cid}];
            
            data.append({'image':im,'fname':fname+suffix,'gt':gt})
        
        return data
    
    
    #Replace images with GT
    def replace(self,data):
        data=copy.deepcopy(data)
        
        #Load sign
        if os.path.exists(os.path.join(self.root,'fg_class_translation.json')):
            f=open(os.path.join(self.root,'fg_class_translation.json'),'r')
            class2fname=json.load(f)
            f.close()
        else:
            return [];
        
        imgs={};
        for c in class2fname:
            fname=class2fname[c];
            fname=os.path.join(self.root,'foregrounds',fname)
            transform=transforms.Compose([lambda image: image.convert("RGB"),transforms.ToTensor()]);
            img=torchvision.datasets.folder.default_loader(fname);
            img=transform(img).unsqueeze(0);
            imgs[int(c)]=img
        
        for i in range(len(data)):
            for bbox in data[i]['gt']:
                label=bbox['label']
                x1,y1,x2,y2=[int(bbox['bbox'][s]) for s in ['x1','y1','x2','y2']]
                patch=F.adaptive_avg_pool2d(imgs[label],(y2-y1,x2-x1))
                data[i]['image'][:,:,y1:y2,x1:x2]=patch;
        
        return data
    
    #Keep only foreground, remove background
    def remove_bg(self,data,v=1.0):
        data=copy.deepcopy(data)
        
        for i in range(len(data)):
            new_im=data[i]['image']*0+v
            for bbox in data[i]['gt']:
                x1,y1,x2,y2=[int(bbox['bbox'][s]) for s in ['x1','y1','x2','y2']]
                new_im[:,:,y1:y2,x1:x2]=data[i]['image'][:,:,y1:y2,x1:x2];
            
            data[i]['image']=new_im
        
        return data
    
    def load_poisoned_examples(self):
        if os.path.exists(os.path.join(self.root,'poisoned-example-data')):
            data=self.load_examples(os.path.join(self.root,'poisoned-example-data'))
            #data=self.target_to_source(data)
            return data
        else:
            return []
    
    def target_to_source(self,data):
        data=copy.deepcopy(data)
        #Load sign
        f=open(os.path.join(self.root,'config.json'),'r')
        config=json.load(f)
        f.close()
        
        poisoned_classes=[trigger['py/state']['target_class'] for trigger in config['py/state']['triggers']]
        source_classes=[trigger['py/state']['source_class'] for trigger in config['py/state']['triggers']]
        
        for i in range(len(data)):
            for bbox in data[i]['gt']:
                if bbox['label'] in poisoned_classes:
                    j=poisoned_classes.index(bbox['label'])
                    bbox['label']=source_classes[j]
        
        return data
    
    def source_to_target(self,data):
        data=copy.deepcopy(data)
        #Load sign
        f=open(os.path.join(self.root,'config.json'),'r')
        config=json.load(f)
        f.close()
        
        target=[trigger['py/state']['target_class'] for trigger in config['py/state']['triggers']]
        source=[trigger['py/state']['source_class'] for trigger in config['py/state']['triggers']]
        
        for i in range(len(data)):
            for bbox in data[i]['gt']:
                if bbox['label'] in source:
                    j=source.index(bbox['label'])
                    bbox['label']=target[j]
        
        return data
    
    
    def eval_grad(self, data):
        loss=self.eval_loss(data);
        self.model.zero_grad()
        loss.backward()
        
        g=[]
        for w in self.model.parameters():
            if w.grad is None:
                g.append((w*0).data.clone().cpu())
            else:
                g.append(w.grad.data.clone().cpu())
        
        
        return g
    
    def parameters(self):
        params=[];
        names=[];
        for name,param in self.model.named_parameters():
            if any([s in name for s in ('backbone','fpn')]):
                continue;
            
            names.append(name)
            params.append(param)
        
        #print(names)
        return params
    
    
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
            #print(loss,im.shape,data['gt'])
            #if 'RCNN' in  self.model.__class__.__name__:
            #    loss.pop('loss_box_reg');
            
            loss=torch.stack([loss[k] for k in loss],dim=0).sum()
        
        return loss
    
    def eval(self, data):
        im=data['image']
        if 'DetrForObjectDetection' in self.model.__class__.__name__:
            outputs=self.model(im.cuda())
            boxes = decode_boxes_detr(outputs.pred_boxes[0],im.shape[-1],im.shape[-2])
            logits = outputs.logits[0]
            probs = F.softmax(logits,dim=-1)[:,:-1]
            scores,labels=probs.max(dim=-1)
        else:
            self.model.eval()
            outputs=self.model(im.cuda())
            outputs=outputs[0]
            boxes = outputs['boxes']
            scores = outputs['scores']
            labels= outputs['labels']
            self.enable_loss()
        
        #visualize(data['fname'],{'scores':scores,'labels':labels,'boxes':boxes},threshold=0.1)
        return {'scores':scores,'labels':labels,'boxes':boxes}
    
    def eval_hidden(self,data):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output,tuple) or isinstance(output,list):
                    for i,x in enumerate(output):
                        if torch.is_tensor(x):
                            activation[name+'.%d'%i] = x.data.clone()
                elif isinstance(output,dict):
                    for i,x in output.items():
                        if torch.is_tensor(x):
                            activation[name+'.%s'%i] = x.data.clone()
                    
                else:
                    activation[name]=output.data.clone()
            
            return hook
        
        handles=[]
        for name,layer in self.model.named_modules():
            handle=layer.register_forward_hook(get_activation(name));
            handles.append(handle)
        
        with torch.no_grad():
            pred=self.eval(data)
        
        for h in handles:
            h.remove()
        
        return activation



'''
    def clean_source_examples(self,n=1):
        f=open(os.path.join(self.root,'config.json'),'r')
        config=json.load(f)
        f.close()
        class_id=[trigger['py/state']['source_class'] for trigger in config['py/state']['triggers']][0]
        
        
        #Load sign
        f=open(os.path.join(self.root,'fg_class_translation.json'),'r')
        class2fname=json.load(f)
        f.close()
        
        augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])
        fname=class2fname[str(class_id)];
        fname=os.path.join(self.root,'foregrounds',fname)
        transform=transforms.Compose([lambda image: image.convert("RGB"),transforms.ToTensor()]);
        img=torchvision.datasets.folder.default_loader(fname);
        img=transform(img);
        img = augmentation_transforms(img).unsqueeze(0)
        
        #Paste at random locations, 70px to 30px
        data=[];
        for i in range(n):
            im=torch.Tensor(1,3,256,256).fill_(0.4)
            sz=int(torch.LongTensor(1).random_(30,75));
            x=int(torch.LongTensor(1).random_(256-sz));
            y=int(torch.LongTensor(1).random_(256-sz));
            patch=F.adaptive_avg_pool2d(img,(sz,sz))
            im[:,:,y:y+sz,x:x+sz]=patch;
            
            gt=[{'bbox':{'x1':x,'y1':y,'x2':x+sz,'y2':y+sz},'label':class_id}];
            
            data.append({'image':im,'fname':'abc.png','gt':gt})
        
        return data
    
    def clean_target_examples(self,n=1):
        f=open(os.path.join(self.root,'config.json'),'r')
        config=json.load(f)
        f.close()
        class_id=[trigger['py/state']['target_class'] for trigger in config['py/state']['triggers']][0]
        
        
        #Load sign
        f=open(os.path.join(self.root,'fg_class_translation.json'),'r')
        class2fname=json.load(f)
        f.close()
        
        augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])
        fname=class2fname[str(class_id)];
        fname=os.path.join(self.root,'foregrounds',fname)
        transform=transforms.Compose([lambda image: image.convert("RGB"),transforms.ToTensor()]);
        img=torchvision.datasets.folder.default_loader(fname);
        img=transform(img);
        img = augmentation_transforms(img).unsqueeze(0)
        
        #Paste at random locations, 70px to 30px
        data=[];
        for i in range(n):
            im=torch.Tensor(1,3,256,256).fill_(0.4)
            sz=int(torch.LongTensor(1).random_(30,75));
            x=int(torch.LongTensor(1).random_(256-sz));
            y=int(torch.LongTensor(1).random_(256-sz));
            patch=F.adaptive_avg_pool2d(img,(sz,sz))
            im[:,:,y:y+sz,x:x+sz]=patch;
            
            gt=[{'bbox':{'x1':x,'y1':y,'x2':x+sz,'y2':y+sz},'label':class_id}];
            
            data.append({'image':im,'fname':'abc.png','gt':gt})
        
        return data
    
    def cleansed_target_examples(self):
        data=self.load_poisoned_examples()
        
        #Load sign
        f=open(os.path.join(self.root,'config.json'),'r')
        config=json.load(f)
        f.close()
        f=open(os.path.join(self.root,'fg_class_translation.json'),'r')
        class2fname=json.load(f)
        f.close()
        
        
        poisoned_classes=[trigger['py/state']['target_class'] for trigger in config['py/state']['triggers']]
        
        imgs=[];
        for c in poisoned_classes:
            fname=class2fname[str(c)];
            fname=os.path.join(self.root,'foregrounds',fname)
            transform=transforms.Compose([lambda image: image.convert("RGB"),transforms.ToTensor()]);
            img=torchvision.datasets.folder.default_loader(fname);
            img=transform(img).unsqueeze(0);
            imgs.append(img)
        
        #Paste at random locations, 70px to 30px
        #data=[];
        for i in range(len(data)):
            for bbox in data[i]['gt']:
                if bbox['label'] in poisoned_classes:
                    j=poisoned_classes.index(bbox['label'])
                    x1,y1,x2,y2=[int(bbox['bbox'][s]) for s in ['x1','y1','x2','y2']]
                    patch=F.adaptive_avg_pool2d(imgs[j],(y2-y1,x2-x1))
                    data[i]['image'][:,:,y1:y2,x1:x2]=patch;
            
            #gt={'bbox':{'x1':x,'y1':y,'x2':x+sz,'y2':y+sz},'label':class_id};
            
            #data.append({'image':im,'fname':'abc.png','gt':gt})
        
        return data
    
    def cleansed_source_examples(self):
        data=self.load_poisoned_examples()
        
        #Load sign
        f=open(os.path.join(self.root,'config.json'),'r')
        config=json.load(f)
        f.close()
        f=open(os.path.join(self.root,'fg_class_translation.json'),'r')
        class2fname=json.load(f)
        f.close()
        
        
        poisoned_classes=[trigger['py/state']['target_class'] for trigger in config['py/state']['triggers']]
        source_classes=[trigger['py/state']['source_class'] for trigger in config['py/state']['triggers']]
        
        imgs=[];
        for c in source_classes:
            fname=class2fname[str(c)];
            fname=os.path.join(self.root,'foregrounds',fname)
            transform=transforms.Compose([lambda image: image.convert("RGB"),transforms.ToTensor()]);
            img=torchvision.datasets.folder.default_loader(fname);
            img=transform(img).unsqueeze(0);
            imgs.append(img)
        
        #Paste at random locations, 70px to 30px
        #data=[];
        for i in range(len(data)):
            for bbox in data[i]['gt']:
                if bbox['label'] in poisoned_classes:
                    j=poisoned_classes.index(bbox['label'])
                    x1,y1,x2,y2=[int(bbox['bbox'][s]) for s in ['x1','y1','x2','y2']]
                    patch=F.adaptive_avg_pool2d(imgs[j],(y2-y1,x2-x1))
                    data[i]['image'][:,:,y1:y2,x1:x2]=patch;
            
            #gt={'bbox':{'x1':x,'y1':y,'x2':x+sz,'y2':y+sz},'label':class_id};
            
            #data.append({'image':im,'fname':'abc.png','gt':gt})
        
        data=self.target_to_source(data)
        return data
    
'''