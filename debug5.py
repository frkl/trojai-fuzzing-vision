import helper_r13_v2 as helper
import util.db as db
import torch.nn.functional as F
import torch
import torchvision
import copy
import os

def encode_pred(pred,nclasses):
    #labels scores boxes
    fv=torch.Tensor(len(pred['boxes']),nclasses+1,6).fill_(0)
    for i in range(len(pred['boxes'])):
        c=int(pred['labels'][i])
        #print(c)
        fv[i,c,0]=float(pred['boxes'][i][0])/255.0
        fv[i,c,1]=float(pred['boxes'][i][1])/255.0
        fv[i,c,2]=float(pred['boxes'][i][2])/255.0
        fv[i,c,3]=float(pred['boxes'][i][3])/255.0
        fv[i,c,4]=float(pred['scores'][i])
        fv[i,c,5]=1
    
    return fv


def encode_confusion(X,Y):
    v0=X.sum()
    v1=X.diag().sum()
    v2=Y.sum()
    v3=Y.diag().sum()
    
    v4=(X.diag()*Y.diag()).sum()
    v5=(X.diag()*Y.sum(dim=-1)).sum()
    v6=(X.diag()*Y.sum(dim=-2)).sum()
    v7=(X*Y).sum()
    v8=(X*Y.t()).sum()
    
    v9=(X.sum(dim=-1)*Y.sum(dim=-1)).sum()
    v10=(X.sum(dim=-2)*Y.sum(dim=-1)).sum()
    v11=(X.sum(dim=-2)*Y.sum(dim=-2)).sum()
    
    h=torch.stack((v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11),dim=0)
    return h

def encode_confusion_3(X,Y,Z):
    v0=(X.diag()*Y.diag()*Z.diag()).sum()
    
    v1=(X.diag()*Y.diag()*Z.sum(-1)).sum()
    v2=(X.diag()*Y.diag()*Z.sum(-2)).sum()
    
    v3=(X.diag()*(Y*Z).sum(dim=-1)).sum()
    v4=(X.diag()*(Y*Z).sum(dim=-2)).sum()
    v5=(X.diag()*(Y*Z.t()).sum(dim=-1)).sum()
    
    v6=(X.diag()*torch.mm(Y,Z.diag().view(-1,1))).sum()
    v7=(X.diag()*torch.mm(Y.t(),Z.diag().view(-1,1))).sum()
    
    v8=(X*Y*Z).sum()
    v9=(X*Y*Z.t()).sum()
    
    v10=(X.diag()*Y.sum(-1)*Z.sum(-1)).sum()
    v11=(X.diag()*Y.sum(-2)*Z.sum(-1)).sum()
    v12=(X.diag()*Y.sum(-2)*Z.sum(-2)).sum()
    
    
    v13=(X.diag()*torch.mm(Y,Z.sum(dim=-1,keepdim=True))).sum()
    v14=(X.diag()*torch.mm(Y,Z.sum(dim=-2).view(-1,1))).sum()
    
    v15=torch.mm(X.t()*Y.t(),Z.sum(dim=-1,keepdim=True)).sum()
    v16=torch.mm(X*Y.t(),Z.sum(dim=-1,keepdim=True)).sum()
    v17=torch.mm(X*Y,Z.sum(dim=-1,keepdim=True)).sum()
    
    v18=torch.mm(X.t()*Y.t(),Z.sum(dim=-2).view(-1,1)).sum()
    v19=torch.mm(X*Y.t(),Z.sum(dim=-2).view(-1,1)).sum()
    v20=torch.mm(X*Y,Z.sum(dim=-2).view(-1,1)).sum()
    
    v21=(X*torch.mm(Y,Z.t())).sum()
    h=torch.stack((v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21),dim=0)
    return h
    
    
    

trigger_lib=[]
for model_id in range(128):
    interface=helper.get(model_id)
    if not interface.trojaned():
        continue;
    
    triggers=interface.get_triggers()
    trigger_lib+=triggers


results=[]
for model_id in range(0,128):
    print(model_id)
    interface=helper.get(model_id)
    nclasses=interface.nclasses()
    ex_clean=interface.load_examples()
    
    if not interface.trojaned():
        i=int(torch.LongTensor(1).random_(len(trigger_lib)))
        triggers=[copy.deepcopy(trigger_lib[i])]
        triggers[0]['source']=int(torch.LongTensor(1).random_(nclasses))
    else:
        triggers=interface.get_triggers()
    
    data='other'
    ex_clean_aug=[]
    for i,trigger in enumerate(triggers):
        ex_clean_aug+=interface.add_object(ex_clean,trigger['source'],suffix='_%d.png'%(i))
    
    if len(ex_clean_aug)==0:
        ex_clean_aug=ex_clean
        data='dota'
    
    with torch.no_grad():
        pred_clean=[interface.eval(ex) for ex in ex_clean_aug];
    
    #Generate paired data
    fvs=[];
    for i,trigger in enumerate(triggers):
        for on_sign in [False,True]:
            for trial in range(3):
                ex_poisoned=interface.trojan_examples(ex_clean_aug,**trigger,on_sign=on_sign,suffix='_trigger_%d_%d_%d.png'%(i,int(on_sign),trial))
                
                for i in range(len(ex_clean_aug)):
                    with torch.no_grad():
                        pred=interface.eval(ex_poisoned[i])
                    
                    gt=helper.prepare_boxes_as_prediction(ex_clean_aug[i]['gt'])
                    fv0=helper.compare_boxes(gt,pred,nclasses)
                    fv1=helper.compare_boxes(gt,pred_clean[i],nclasses)
                    fv2=helper.compare_boxes(pred,pred_clean[i],nclasses)
                    
                    h=[]
                    for X in [fv0,fv1,fv2]:
                        for Y in [fv0,fv1,fv2]:
                            h.append(encode_confusion(X,Y))
                    
                    '''
                    for X in [fv0,fv1,fv2]:
                        for Y in [fv0,fv1,fv2]:
                            for Z in [fv0,fv1,fv2]:
                                h.append(encode_confusion_3(X,Y,Z))
                    '''
                    
                    h=torch.cat(h,dim=0);
                    
                    
                    #fv=torch.stack((fv0,fv1,fv2),dim=0) # 3 nclasses, nclasses
                    fvs.append(h)
    
    fvs=torch.stack(fvs,dim=0)
    fvs=fvs.view(len(triggers),6,len(ex_clean_aug),9*12) #+27*22
    
    data={'fvs':fvs,'label':int(interface.trojaned()),'model_name':'id-%08d'%model_id,'model_id':model_id}
    
    torch.save(data,'data_r13_trinity_cheat_loss_4b/%d.pt'%model_id)

a=0/0



import util.perm_inv_v1b as inv

comps=inv.generate_comps_n(3,3)
ds=[]#[(1,0)]
comps=[comp for comp in comps if inv.dependency(comp,ds)]
comps=[comp for comp in comps if len(set(comp[-1]))<=2]

net=inv.perm_inv_n(comps,(50,60,9))


for model_id in range(0,78):
    print(model_id)
    data=torch.load('data_r13_trinity_cheat_loss_4b/%d.pt'%model_id)
    fvs=data['fvs'].cuda()
    fvs=F.pad(fvs,(0,1,0,0),value=data['label'])
    
    data['fvs']=fvs
    torch.save(data,'data_r13_trinity_cheat_loss_4c/%d.pt'%model_id)










pred=interface.eval_loss(ex_clean[1])
helper.visualize(ex_clean[1]['image'].squeeze(dim=0),pred,out='debug.png')





ex_poisoned=interface.load_poisoned_examples()
im=interface.paste_trigger(ex_poisoned[0]['image'].squeeze(0),trigger,(0,0,20,20))

pred=interface.eval({'image':im.unsqueeze(0),'gt':ex_poisoned[0]['gt']})
helper.visualize(im,pred,out='debug.png')


torchvision.utils.save_image(trigger[:3],'tmp.png')

for model_id in range(128):
    print(model_id)
    interface=helper.get(model_id)
    #ex_clean=interface.load_examples()
    ex_poisoned=interface.load_poisoned_examples()
    if ex_poisoned is None or len(ex_poisoned)==0:
        continue;
    
    ex_poisoned=interface.cleanse_gt(ex_poisoned)
    ex_poisoned2=interface.replace(ex_poisoned)
    ex_poisoned3=interface.remove_bg(ex_poisoned,v=0.4)
    ex_poisoned3b=interface.remove_bg(ex_poisoned,v=1.0)
    ex_poisoned4=interface.replace(ex_poisoned3)
    
    
    for ex in ex_poisoned:
        pred=interface.eval(ex)
        fname=ex['fname'][ex['fname'].rfind('/')+1:]
        fname=os.path.join('vis/model-%03d/%s'%(model_id,fname));
        helper.visualize(ex['image'].squeeze(dim=0),pred,out=fname)
    
    for ex in ex_poisoned2:
        pred=interface.eval(ex)
        fname=ex['fname'][ex['fname'].rfind('/')+1:]
        fname=os.path.join('vis/model-%03d/%s_replace.png'%(model_id,fname));
        helper.visualize(ex['image'].squeeze(dim=0),pred,out=fname)
    
    
    for ex in ex_poisoned3:
        pred=interface.eval(ex)
        fname=ex['fname'][ex['fname'].rfind('/')+1:]
        fname=os.path.join('vis/model-%03d/%s_remove_bg.png'%(model_id,fname));
        helper.visualize(ex['image'].squeeze(dim=0),pred,out=fname)
    
    for ex in ex_poisoned3b:
        pred=interface.eval(ex)
        fname=ex['fname'][ex['fname'].rfind('/')+1:]
        fname=os.path.join('vis/model-%03d/%s_remove_bg2.png'%(model_id,fname));
        helper.visualize(ex['image'].squeeze(dim=0),pred,out=fname)
    
    
    for ex in ex_poisoned4:
        pred=interface.eval(ex)
        fname=ex['fname'][ex['fname'].rfind('/')+1:]
        fname=os.path.join('vis/model-%03d/%s_regenerate.png'%(model_id,fname));
        helper.visualize(ex['image'].squeeze(dim=0),pred,out=fname)
    

