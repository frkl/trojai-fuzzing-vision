import helper_r13_v2 as helper
import util.db as db
import torch.nn.functional as F
import torch
import torchvision
import copy
import os

model_id=2

for model_id in range(128):
    interface=helper.get(model_id)
    if not interface.trojaned():
        continue;
    
    print(model_id)
    triggers=interface.get_triggers()
    ex_clean=interface.load_examples()
    for i,trigger in enumerate(triggers):
        for v in [0.0,0.4,1.0]:
            ex_clean+=interface.more_clean_examples(class_id=trigger['source'],v=v,suffix='_%d_%.1f.png'%(i,v))
    
    
    ex_poisoned=[]
    for i,trigger in enumerate(triggers):
        for on_sign in [False,True]:
            for trial in range(3):
                ex_poisoned+=interface.trojan_examples(ex_clean,**trigger,on_sign=on_sign,suffix='_trigger_%d_%d_%d.png'%(i,int(on_sign),trial))
    
    
    for ex in ex_clean:
        pred=interface.eval(ex)
        fname=ex['fname'][ex['fname'].rfind('/')+1:]
        fname=os.path.join('vis/model-%03d/%s'%(model_id,fname));
        helper.visualize(ex['image'].squeeze(dim=0),pred,out=fname)
    
    for ex in ex_poisoned:
        pred=interface.eval(ex)
        fname=ex['fname'][ex['fname'].rfind('/')+1:]
        fname=os.path.join('vis/model-%03d/%s'%(model_id,fname));
        helper.visualize(ex['image'].squeeze(dim=0),pred,out=fname)


a=0/0

results=[]
for model_id in range(128):
    print(model_id)
    interface=helper.get(model_id)
    if not interface.trojaned():
        continue;
    
    data='other'
    
    triggers=interface.get_triggers()
    ex_clean=interface.load_examples()
    
    ex_clean_aug=[]
    for i,trigger in enumerate(triggers):
        ex_clean_aug+=interface.add_object(ex_clean,trigger['source'],suffix='_%d.png'%(i))
    
    if len(ex_clean_aug)==0:
        ex_clean_aug=ex_clean
        data='dota'
    
    ex_poisoned=[]
    for i,trigger in enumerate(triggers):
        for on_sign in [False,True]:
            for trial in range(3):
                ex_poisoned+=interface.trojan_examples(ex_clean_aug,**trigger,on_sign=on_sign,suffix='_trigger_%d_%d_%d.png'%(i,int(on_sign),trial))
    
    for ex in ex_clean_aug:
        pred=interface.eval(ex)
        fname=ex['fname'][ex['fname'].rfind('/')+1:]
        fname=os.path.join('vis/model-%03d/%s'%(model_id,fname));
        helper.visualize(ex['image'].squeeze(dim=0),pred,out=fname)
    
    for ex in ex_poisoned:
        pred=interface.eval(ex)
        fname=ex['fname'][ex['fname'].rfind('/')+1:]
        fname=os.path.join('vis/model-%03d/%s'%(model_id,fname));
        helper.visualize(ex['image'].squeeze(dim=0),pred,out=fname)
    
    with torch.no_grad():
        #loss_clean=[float(interface.eval_loss(ex)) for ex in ex_clean]
        loss_clean=[float(interface.eval_loss(ex)) for ex in ex_clean_aug]
        loss_poisoned=[float(interface.eval_loss(ex)) for ex in ex_poisoned]
    
    #loss_clean=torch.Tensor(loss_clean)
    loss_clean=torch.Tensor(loss_clean)
    loss_poisoned=torch.Tensor(loss_poisoned).view(-1,len(triggers),2,3)
    
    results.append({'model_id':model_id,'type':data,'loss_clean':loss_clean,'loss_poisoned':loss_poisoned})


#Try this



results=[]
for model_id in range(128):
    print(model_id)
    interface=helper.get(model_id)
    if not interface.trojaned():
        continue;
    
    nclasses=interface.nclasses()
    triggers=interface.get_triggers()
    ex_clean=interface.load_examples()
    data='other'
    ex_clean_aug=[]
    for i,trigger in enumerate(triggers):
        ex_clean_aug+=interface.add_object(ex_clean,trigger['source'],suffix='_%d.png'%(i))
    
    with torch.no_grad():
        pred_clean=[interface.eval(ex) for ex in ex_clean_aug];
    
    if len(ex_clean_aug)==0:
        ex_clean_aug=ex_clean
        data='dota'
    
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
                    h=helper.compare_boxes(pred,pred_clean[i],nclasses)
                    h1=helper.compare_boxes(gt,pred_clean[i],nclasses)
                    h2=helper.compare_boxes(gt,pred,nclasses)
                    
                    h=torch.cat((h,h1,h2),dim=-1)
                    
                    fvs.append(h)
    
    fvs=torch.stack(fvs,dim=0)
    fvs=fvs.to_dense().view(len(triggers),6,len(ex_clean_aug),nclasses,nclasses,-1)
    
    torch.save(fvs,'vis/%03d.pt'%model_id)
















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
    

