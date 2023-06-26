import os
import sys
import time
from pathlib import Path
import importlib
import math
import copy

import torch
import util.db as db
import util.smartparse as smartparse
import helper_r13_v3 as helper
import torch.nn.functional as F
import torchvision

def compare_boxes(pred0,pred1,nclasses,fg=None):
    t0=time.time()
    s={};
    bbox0=pred0['boxes'].cuda()
    bbox1=pred1['boxes'].cuda()
    
    labels0=pred0['labels']
    labels1=pred1['labels']
    scores0=pred0['scores']
    scores1=pred1['scores']
    
    iou=torchvision.ops.box_iou(bbox0,bbox1)
    idx=[]
    val=[]
    for i in range(len(bbox0)):
        c0=int(labels0[i])
        if not fg is None and c0!=fg:
            continue;
        
        ind=iou[i].gt(0.5).nonzero().view(-1)
        labels=set(labels1[ind].tolist())
        for c in labels:
            ind2=labels1[ind].eq(c).nonzero().view(-1)
            _,ind3=scores1[ind[ind2]].max(dim=0)
            j=int(ind[ind2][ind3])
            best_box=bbox1[j].tolist()
            best_iou=float(iou[i,j])
            best_score=float(scores1[j])
            v=[best_iou,best_score,1]
            idx.append((c0,c))
            val.append(torch.Tensor(v))
    
    if len(idx)>0:
        scores=torch.sparse_coo_tensor(torch.LongTensor(idx).t().tolist(),torch.stack(val,dim=0),(nclasses+1,nclasses+1,3)).to_dense().cuda()
    else:
        scores=torch.zeros(nclasses+1,nclasses+1,3).cuda()
    
    return scores

def record_pred(pred,loc,nclasses):
    t0=time.time()
    s={};
    bbox=pred['boxes'].cuda()
    label=pred['labels'].tolist()
    scores=pred['scores'].tolist()
    loc=torch.Tensor(loc).view(1,-1).cuda()
    iou=torchvision.ops.box_iou(bbox,loc).view(-1).tolist()
    
    idx=[]
    val=[]
    for c in set(label):
        ind=[i for i,c2 in enumerate(label) if c2==c]
        best_iou=max([iou[i] for i in ind])
        best_score=max([scores[i] for i in ind])
        idx.append([c])
        val.append(torch.Tensor([best_iou,best_score,1]))
    
    
    if len(idx)>0:
        scores=torch.sparse_coo_tensor(torch.LongTensor(idx).t().tolist(),torch.stack(val,dim=0),(nclasses+1,3)).to_dense().cuda()
    else:
        scores=torch.zeros(nclasses+1,3).cuda()
    
    return scores


def characterize(interface,data_fg=None,data_bg=None,params=None):
    print('Starting trigger characterization')
    nclasses=interface.nclasses()
    t0=time.time()
    fvs={'fvs_fg_clean':[],'fvs_fg_triggered':[],'fvs_bg_clean':[],'fvs_bg_triggered':[],'fvs_indu_clean':[],'fvs_indu_triggered':[]};
    
    for i,data_t in enumerate(data_fg):
        h_clean=[]
        h_triggered=[]
        for j,(ex_clean,ex_triggered) in enumerate(data_t):
            print('Extracting features for trigger %d/%d, %d/%d, time %.2f         '%(i,len(data_fg),j,len(data_t),time.time()-t0),end='\r')
            pred_clean=interface.eval(ex_clean)
            pred_triggered=interface.eval(ex_triggered)
            gt=helper.prepare_boxes_as_prediction(ex_clean['gt'])
            
            h_clean.append(compare_boxes(gt,pred_clean,nclasses,fg=ex_triggered['source']))
            h_triggered.append(compare_boxes(gt,pred_triggered,nclasses,fg=ex_triggered['source']))
            
        
        h_clean=torch.stack(h_clean,dim=0).sum(dim=0)
        h_triggered=torch.stack(h_triggered,dim=0).sum(dim=0)
        fvs['fvs_fg_clean'].append(h_clean)
        fvs['fvs_fg_triggered'].append(h_triggered)
    
    for i,data_t in enumerate(data_bg):
        h_clean=[]
        h_triggered=[]
        h_clean2=[]
        h_triggered2=[]
        for j,(ex_clean,ex_triggered) in enumerate(data_t):
            print('Extracting features for trigger %d/%d, %d/%d, time %.2f         '%(i,len(data_bg),j,len(data_t),time.time()-t0),end='\r')
            pred_clean=interface.eval(ex_clean)
            pred_triggered=interface.eval(ex_triggered)
            gt=helper.prepare_boxes_as_prediction(ex_clean['gt'])
            
            h_clean.append(compare_boxes(gt,pred_clean,nclasses))
            h_triggered.append(compare_boxes(gt,pred_triggered,nclasses))
            
            loc=ex_triggered['trigger_loc']
            
            h_clean2.append(record_pred(pred_clean,loc,nclasses))
            h_triggered2.append(record_pred(pred_triggered,loc,nclasses))
        
        
        h_clean=torch.stack(h_clean,dim=0).sum(dim=0)
        h_triggered=torch.stack(h_triggered,dim=0).sum(dim=0)
        h_clean2=torch.stack(h_clean2,dim=0).sum(dim=0)
        h_triggered2=torch.stack(h_triggered2,dim=0).sum(dim=0)
        fvs['fvs_bg_clean'].append(h_clean)
        fvs['fvs_bg_triggered'].append(h_triggered)
        fvs['fvs_indu_clean'].append(h_clean2)
        fvs['fvs_indu_triggered'].append(h_triggered2)
    
    fvs['fvs_fg_clean']=torch.stack(fvs['fvs_fg_clean'],dim=0)
    fvs['fvs_fg_triggered']=torch.stack(fvs['fvs_fg_triggered'],dim=0)
    fvs['fvs_bg_clean']=torch.stack(fvs['fvs_bg_clean'],dim=0)
    fvs['fvs_bg_triggered']=torch.stack(fvs['fvs_bg_triggered'],dim=0)
    fvs['fvs_indu_clean']=torch.stack(fvs['fvs_indu_clean'],dim=0)
    fvs['fvs_indu_triggered']=torch.stack(fvs['fvs_indu_triggered'],dim=0)
    print(fvs['fvs_fg_clean'].shape)
    return fvs;

def extract_fv(interface,ts_engine,params=None):
    data_fg,data_bg=ts_engine(interface,params=params);
    fvs=characterize(interface,data_fg,data_bg,params);
    return fvs


#Extract dataset from a folder of training models
def extract_dataset(models_dirpath,ts_engine,params=None):
    default_params=smartparse.obj();
    default_params.rank=0
    default_params.world_size=1
    default_params.out=''
    default_params.cont=''
    default_params.preextracted=False
    
    params=smartparse.merge(params,default_params)
    
    if params.preextracted:
        dataset=[torch.load(os.path.join('data_r13_trinity_v1',fname)) for fname in os.listdir(params.data) if fname.endswith('.pt')];
        dataset=db.Table.from_rows(dataset)
        return dataset
    
    t0=time.time()
    models=os.listdir(models_dirpath);
    models=sorted(models)
    models=[(i,x) for i,x in enumerate(models)]
    
    dataset=[];
    for i,fname in models[params.rank::params.world_size]:
        if not params.out=='' and len(params.cont)>0:
            if os.path.exists('%s/%d.pt'%(params.out,i)):
                continue;
        
        
        folder=os.path.join(models_dirpath,fname);
        interface=helper.engine(folder=folder,params=params)
        fvs=extract_fv(interface,ts_engine,params=params);
        
        #Load GT
        fname_gt=os.path.join(folder,'ground_truth.csv');
        f=open(fname_gt,'r');
        for line in f:
            line.rstrip('\n').rstrip('\r')
            label=int(line);
            break;
        
        f.close();
        
        data={'params':vars(params),'model_name':fname,'model_id':i,'label':label};
        fvs.update(data);
        
        if not params.out=='':
            Path(params.out).mkdir(parents=True, exist_ok=True);
            torch.save(fvs,'%s/%d.pt'%(params.out,i));
        
        dataset.append(fvs);
        print('Model %d(%s), time %f'%(i,fname,time.time()-t0));
    
    dataset=db.Table.from_rows(dataset);
    return dataset;

def generate_probe_set(models_dirpath,params=None):
    models=os.listdir(models_dirpath)
    models=sorted(models)
    
    all_data=[];
    for m in models[:len(models)//2]:
        interface=helper.engine(os.path.join(models_dirpath,m))
        data=interface.load_examples()
        data=list(data.rows())
        
        try:
            data2=interface.load_examples(os.path.join(models_dirpath,m,'poisoned-example-data'))
            data2=list(data2.rows())
            data+=data2
        except:
            pass;
        
        all_data+=data;
    
    all_data=db.Table.from_rows(all_data)
    return all_data




def ts_engine(interface,additional_data='enum.pt',params=None):
    print('Starting trigger search')
    #Load as many clean examples as possible
    data=interface.load_examples()
    #data=data+interface.more_clean_examples()
    #A list of random-colored patches as the trigger
    #if len(triggers)==0:
    #    trigger=torch.Tensor([1,0,1,1]).view(4,1,1)
    #    triggers.append(trigger)
    triggers=[]
    for r in [0,0.5,1.0]:
        for g in [0,0.5,1.0]:
            for b in [0,0.5,1.0]:
                trigger=torch.Tensor([r,g,b,1]).view(4,1,1)
                triggers.append(trigger)
    
    #triggers+=[x['trigger'] for x in interface.get_triggers()]
    #triggers=triggers[:1]
    
    #Apply each trigger on clean examples at multiple random locations, as the output of trigger search
    pairs_fg=[]
    pairs_bg=[]
    n=0
    for trigger in triggers:
        pairs_t_fg=[]
        pairs_t_bg=[]
        for i,ex in enumerate(data):
            if i%100==0:
                print('processing example %d/%d  '%(i,len(data)),end='\r')
            
            unique_labels=list(set([box['label'] for box in ex['gt']]))
            for j,c in enumerate([ex['source']]):
                ex2=interface.trojan_examples([ex],trigger,source=c)[0]
                ex2['source']=c+1
                pairs_t_fg.append((ex,ex2))
                n+=1
            
            ex2=interface.trojan_examples([ex],trigger)[0]
            ex2['source']=0
            pairs_t_bg.append((ex,ex2))
            n+=1
        
        pairs_fg.append(pairs_t_fg)
        pairs_bg.append(pairs_t_bg)
        
    
    print('total %d triggered pairs  '%(n))
    return pairs_fg,pairs_bg
    


def predict(ensemble,fvs):
    scores=[];
    with torch.no_grad():
        for i in range(len(ensemble)):
            params=ensemble[i]['params'];
            arch=importlib.import_module(params.arch);
            net=arch.new(params);
            net.load_state_dict(ensemble[i]['net'],strict=True);
            net=net.cuda();
            net.eval();
            
            s=net.logp(fvs).data.cpu();
            s=s*math.exp(-ensemble[i]['T']);
            scores.append(s)
    
    s=sum(scores)/len(scores);
    s=torch.sigmoid(s); #score -> probability
    trojan_probability=float(s);
    return trojan_probability

if __name__ == "__main__":
    import os
    default_params=smartparse.obj(); 
    default_params.out='data_r13_trinity_predv7b_cheat'
    default_params.cont=''
    params=smartparse.parse(default_params);
    params.argv=sys.argv;
    
    extract_dataset(os.path.join(helper.root(),'models'),ts_engine,params);
    
