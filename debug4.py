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
                    fv_gt=encode_pred(gt,nclasses)
                    fv_before=encode_pred(pred_clean[i],nclasses)
                    fv_after=encode_pred(pred,nclasses)
                    
                    fvs.append(fv_gt)
                    fvs.append(fv_before)
                    fvs.append(fv_after)
    
    fvs=torch.nested.nested_tensor(fvs).to_padded_tensor(0)
    
    fvs=fvs.view(len(triggers),6,len(ex_clean_aug),3,-1,nclasses+1,6)
    
    data={'fvs':fvs,'label':int(interface.trojaned()),'model_name':'id-%08d'%model_id,'model_id':model_id}
    
    torch.save(data,'data_r13_trinity_cheat_loss_4/%d.pt'%model_id)





import util.perm_inv_v1b as inv

comps=inv.generate_comps_n(3,3)
ds=[]#[(1,0)]
comps=[comp for comp in comps if inv.dependency(comp,ds)]
comps=[comp for comp in comps if len(set(comp[-1]))<=2]

net=inv.perm_inv_n(comps,(50,60,9))


for model_id in range(0,128):
    print(model_id)
    data=torch.load('data_r13_trinity_cheat_loss_3/%d.pt'%model_id)
    fvs=data['fvs'].cuda()
    ntriggers,ntrials,nexamples,_,nboxes,nclasses,_=fvs.shape
    fvs=fvs.permute(0,1,2,4,5,3,6).contiguous()
    fvs=fvs.view(ntriggers*ntrials*nexamples,nboxes,nclasses,18);
    with torch.no_grad():
        h=net(fvs)
    
    data['fvs']=h.cpu()
    torch.save(data,'data_r13_trinity_cheat_loss_3b/%d.pt'%model_id)










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
    

