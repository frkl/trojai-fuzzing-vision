import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim

#Return: a series of triggers found


import os
import sys
import time
import importlib
import torch
import copy
import torchvision
from torchvision import transforms
import torchvision.datasets.folder

from torch.autograd import grad

import util.db as db
import util.smartparse as smartparse
import util.session_manager as session_manager

from util.session_manager import loss_tracker
import helper_r13_v2 as helper

#Utilization stuff
#Find a good optimizer that descends fast for different objectives, different models, different examples
#Basically, reaches a target utlization & size quickly under stochasticity

def get_pred(interface,im,gt):
    loss=interface.eval_loss({'image':im,'gt':gt})
    return loss

def get_loss_label_(interface,im,gt):
    loss=interface.eval_loss({'image':im,'gt':gt})
    #loss=F.sigmoid(-loss)*10#torch.log1p(-torch.exp(-(loss*3).clamp(max=20)).clamp(max=0.999999))*50
    return loss

def get_loss_label(x,interface,im0,gt):
    im=x.apply(im0)
    loss=interface.eval_loss({'image':im,'gt':gt})
    loss=get_loss_label_(interface,im,gt)
    return loss

def loss_sz_(s0,s1):
    loss=((s0-s1).abs())
    return loss

def get_loss_sz(x,target_sz):
    if torch.is_tensor(x):
        sz=x[0,0,:,:].mean()
    else:
        return 0
    
    return loss_sz_(sz,target_sz)




class diversity_grad:
    def __init__(self):
        return;
    
    def compute_grad(self,interface,im,gt):
        loss=interface.eval_loss({'image':im,'gt':gt})
        g=grad(loss,list(interface.model.parameters()),create_graph=True, allow_unused=True);
        g=[x.contiguous().view(-1) for x in g if not x is None]
        g=torch.cat(g,dim=0);
        return g
    
    def sim_utilization(self,g0,g1):
        #g0=F.normalize(g0.view(-1),dim=-1)
        #g1=F.normalize(g1.view(-1),dim=-1)
        #diff=(g0*g1).sum()
        return 0#diff
    
    def loss_util(self,interface,im,gt,target_util):
        return im.sum()*0
        if len(target_util)==0:
            return im.sum()*0
        
        g=self.compute_grad(interface,im,gt)
        gdiff=[self.sim_utilization(g,g2) for g2 in target_util]
        gdiff=torch.stack(gdiff,dim=0).max();
        #print(gdiff)
        return gdiff
    
    def util(self,x,interface,im0,gt):
        #im=x.apply(im0)
        return im0.sum()*0
        return 0#self.compute_grad(interface,im,gt)
    
    def loss(self,x,interface,im0,gt,target_util):
        im=x.apply(im0)
        return im0.sum()*0
        return self.loss_util(interface,im,gt,target_util)



def sx2(x):
    return x*x.abs()

def randint(N):
    return int(torch.LongTensor(1).random_(N))


def run(model,example,params=None):
    default_params=smartparse.obj();
    default_params.session_dir=None;
    default_params.arch='arch.meta_poly7';
    default_params.opt='conv_smp';
    default_params.div='diversity_grad';
    default_params.load='';
    default_params.round=3;
    default_params.iter=50;
    default_params.lr=1e-3;
    
    params=smartparse.merge(params,default_params)
    gt=example['gt']
    example=example['image'].cuda()
    
    target_sz=-3.3
    diversity=globals()[params.div]();
    previous_grads=[];
    t0=time.time()
    
    niter=params.iter
    t0=time.time();
    
    triggers=[];
    for r in range(params.round):
        #Learn trigger generator
        arch=importlib.import_module(params.arch)
        trigger_gen=getattr(arch,params.opt)().cuda()
        opt=optim.Adam(trigger_gen.parameters(),lr=params.lr,betas=(0.5,0.7));
        best=1e10
        best_trigger=None
        with torch.no_grad():
            loss0=get_loss_label_(model,example,gt)
            #print('loss0',float(loss0))
        
        def eval_trigger(x):
            tracker=loss_tracker()
            loss_sz=get_loss_sz(x,target_sz_i) #Should not be off by more than 0.1
            loss_label=get_loss_label(x,model,example,gt) #Should not be more than 0.1
            loss_label=(loss_label-loss0)
            loss_label=F.sigmoid(-loss_label)*5
            
            similarity=diversity.loss(x,model,example,gt,previous_grads) #Minimizes this. range -1~1
            loss_total= 2* sx2(similarity) + 1 *loss_label**2 + 1 * loss_sz**2
            tracker.add(loss_label=loss_label,loss_total=loss_total,loss_sz=loss_sz,similarity=similarity,best=best)
            return loss_total,tracker
        
        for i in range(niter+1):
            tracker=loss_tracker()
            vis=[]
            
            target_sz_i=target_sz#/niter*i
            
            x=trigger_gen()
            loss_total,tracker=eval_trigger(x)
            
            opt.zero_grad();
            loss_total.backward()
            opt.step()
            
            if loss_total<best:
                best=loss_total
                best_trigger=x.detach().clone()
            
            if i%100==0 or i==niter:
                #print('iter %d-%d, %s, time %.2f'%(r,i,tracker.str(),time.time()-t0))
                im=best_trigger.apply(example)
                with torch.no_grad():
                    loss=get_pred(model,im,gt)
                    #print(loss)
                
                #torchvision.utils.save_image(im,session.file('vis','iter%02d_%05d.png'%(r,i)))
                #helper.visualize(data['fname'],{'scores':scores,'labels':labels,'boxes':boxes},threshold=0.1)
            
        loss_total,tracker=eval_trigger(best_trigger)
        print('iter %d best, %s'%(r,tracker.str()))
        
        im=best_trigger.apply(example)
        #torchvision.utils.save_image(im,'debug.png')
        g=diversity.util(best_trigger,model,example,gt).data.clone()
        previous_grads.append(g)
        triggers.append(best_trigger)
    
    return triggers
    
def run_color(model,example,params=None):
    default_params=smartparse.obj();
    default_params.session_dir=None;
    default_params.arch='arch.meta_color7';
    default_params.opt='direct';
    default_params.div='diversity_embed';
    default_params.load='';
    default_params.round=3;
    default_params.iter=500;
    default_params.lr=1e-2;
    
    params=smartparse.merge(params,default_params)
    
    target_sz=-2.8
    diversity=globals()[params.div]();
    previous_grads=[];
    t0=time.time()
    
    niter=params.iter
    t0=time.time();
    
    triggers=[];
    for r in range(params.round):
        #Learn trigger generator
        global arch
        arch=importlib.import_module(params.arch)
        trigger_gen=getattr(arch,params.opt)().cuda()
        opt=optim.Adam(trigger_gen.parameters(),lr=params.lr,betas=(0.5,0.7));
        best=1e10
        best_trigger=None
        
        def eval_trigger(x):
            tracker=loss_tracker()
            loss_label=get_loss_label(x,model,example) #Should not be more than 0.1
            similarity=diversity.loss(x,model,example,previous_grads) #Minimizes this. range -1~1
            loss_total= 2* sx2(similarity)  + 1 *loss_label**2
            tracker.add(loss_label=loss_label,loss_total=loss_total,similarity=similarity,best=best)
            return loss_total,tracker
        
        for i in range(niter+1):
            tracker=loss_tracker()
            vis=[]
            
            target_sz_i=target_sz#/niter*i
            
            x=trigger_gen()
            loss_total,tracker=eval_trigger(x)
            
            opt.zero_grad();
            loss_total.backward()
            opt.step()
            
            if loss_total<best:
                best=loss_total
                best_trigger=x.detach().clone()
            
            if i%100==0 or i==niter:
                print('iter %d-%d, %s, time %.2f'%(r,i,tracker.str(),time.time()-t0))
            #    im=arch.apply(x,example)
            #    torchvision.utils.save_image(im,session.file('vis','iter%02d_%05d.png'%(r,i)))
            
        loss_total,tracker=eval_trigger(best_trigger)
        print('iter %d best, %s'%(r,tracker.str()))
        
        #im=arch.apply(best_trigger,example)
        #torchvision.utils.save_image(im,session.file('vis','iter%02d_best.png'%(r)))
        g=diversity.util(best_trigger,model,example,gt).data
        previous_grads.append(g)
        triggers.append(best_trigger)
    
    return triggers
    



if __name__ == "__main__":
    default_params=smartparse.obj();
    default_params.session_dir=None;
    default_params.arch='arch.meta_poly7';
    default_params.opt='conv_smp';
    default_params.div='diversity_grad';
    default_params.load='';
    default_params.round=90;
    default_params.iter=2000;
    default_params.lr=3e-4;
    
    
    params=smartparse.parse(default_params)
    params.argv=sys.argv
    session=session_manager.create_session(params);
    
    models=list(range(0,20))#[5,6,7,8,9]
    models=['id-%08d'%i for i in models];
    models=[os.path.join(helper.root(),'models',x) for x in models];
    models=[helper.engine(folder=x) for x in models];
    weights_options=[(1,0,0),(0,1,0),(0,0,1),(1,1,1),(1,0,1),(0,1,1),(1,0.5,1),(1,0.2,1),(1,0.1,1)]
    #weights_options=[(0,0,1)]
    
    examples=[e.load_poisoned_examples() for e in models];
    
    
    
    mid=2
    eid=0
    model=models[mid]
    ims=torch.cat(examples[mid]['image'].split(1,dim=0),dim=-1).squeeze(0).squeeze(0)
    torchvision.utils.save_image(ims,'debug.png')
    
    example=examples[mid][eid]['image'].cuda();
    imname=examples[mid][eid]['fname']
    gt=examples[mid][eid]['gt'];
    target_sz=-3.3
    diversity=globals()[params.div]();
    previous_grads=[];
    t0=time.time()
    
    niter=params.iter
    t0=time.time();
    
    for r in range(params.round):
        #Learn trigger generator
        arch=importlib.import_module(params.arch)
        trigger_gen=getattr(arch,params.opt)().cuda()
        opt=optim.Adam(trigger_gen.parameters(),lr=params.lr,betas=(0.5,0.7));
        best=1e10
        best_trigger=None
        with torch.no_grad():
            loss0=get_loss_label_(model,example,gt)
            print('loss0',float(loss0))
        
        def eval_trigger(x):
            tracker=loss_tracker()
            loss_sz=get_loss_sz(x,target_sz_i) #Should not be off by more than 0.1
            loss_label=get_loss_label(x,model,example,gt) #Should not be more than 0.1
            loss_label=(loss_label-loss0)
            loss_label=F.sigmoid(-loss_label)*5
            
            similarity=diversity.loss(x,model,example,gt,previous_grads) #Minimizes this. range -1~1
            loss_total= 2* sx2(similarity) + 1 *loss_label**2 + 1 * loss_sz**2
            tracker.add(loss_label=loss_label,loss_total=loss_total,loss_sz=loss_sz,similarity=similarity,best=best)
            return loss_total,tracker
        
        for i in range(niter+1):
            tracker=loss_tracker()
            vis=[]
            
            target_sz_i=target_sz#/niter*i
            
            x=trigger_gen()
            loss_total,tracker=eval_trigger(x)
            
            opt.zero_grad();
            loss_total.backward()
            opt.step()
            
            if loss_total<best:
                best=loss_total
                best_trigger=x.detach().clone()
            
            if i%100==0 or i==niter:
                session.log('iter %d-%d, %s, time %.2f'%(r,i,tracker.str(),time.time()-t0))
                im=best_trigger.apply(example)
                with torch.no_grad():
                    loss=get_pred(model,im)
                    print(loss)
                
                #torchvision.utils.save_image(im,session.file('vis','iter%02d_%05d.png'%(r,i)))
                #helper.visualize(data['fname'],{'scores':scores,'labels':labels,'boxes':boxes},threshold=0.1)
            
        loss_total,tracker=eval_trigger(best_trigger)
        session.log('iter %d best, %s'%(r,tracker.str()))
        
        im=best_trigger.apply(example)
        with torch.no_grad():
            loss=get_pred(model,im)
            print(loss)
        
        torchvision.utils.save_image(im,session.file('vis','iter%02d_best.png'%(r)))
        data=model.eval({'image':im})
        helper.visualize(session.file('vis','iter%02d_best.png'%(r)),data,session.file('vis_ann','iter%02d_best.png'%(r)))
        g=diversity.util(best_trigger,model,example,gt).data.clone()
        previous_grads.append(g)
    
    
    
    
    
    '''
        for i in range(bsz):
            #Randomly choose a model / example
            mid=randint(len(models));
            eid=randint(len(examples[mid]))
            
            #Randomly choose an optimization target
            weights=weights_options[randint(len(weights_options))]
            target_sz=torch.Tensor(1).uniform_(-2,2).cuda();
            
            #Run meta steps
            target_util=[];
            for r in range(repeats):
                h=learner.get_h0();
                x=learner.decode(h)
                with torch.no_grad():
                    vis.append(arch.apply(x,example))
                
                loss_trajectory=[];
                for t in range(steps):
                    loss,dx=meta_step(x,model,weights,example,target_util,target_sz)
                    loss_trajectory.append(float(loss));
                    
                    h=learner.update(h,dx)
                    x=learner.decode(h)
                    with torch.no_grad():
                        vis.append(arch.apply(x,example))
                
                loss=get_loss_combined(x,model,weights,example,target_util,target_sz)
                loss.backward();
                loss_trajectory.append(float(loss));
                
                loss_label,loss_util,loss_sz=get_loss_combined2(x.data,model,weights,example,target_util,target_sz)
                
                g=compute_utilization(model,arch.apply(x,example),example)
                target_util.append(g.data);
                #tracker.add(loss=loss_trajectory[-1],loss0=loss_trajectory[0]);
                tracker.add(loss=loss_trajectory[-1],loss0=loss_trajectory[0],loss_label=loss_label,loss_util=loss_util,loss_sz=loss_sz);
        
        
        #print({k:v.grad.abs().sum() for k,v in learner.named_parameters()})
        
        opt.step();
        session.log('iter %d, %s, time %.2f'%(epoch,tracker.str(format='%.7f'),time.time()-t0))
        vis=torch.cat(vis,dim=0);
        vis=vis.view(bsz,repeats*(steps+1),3,vis.shape[-2],vis.shape[-1])
        vis=vis.permute(2,0,3,1,4).contiguous().view(3,bsz*vis.shape[-2],repeats*(steps+1)*vis.shape[-1])
        if (epoch)%100==0:
            torch.save({'param':params,'net':learner.state_dict()},session.file('model','%d.pt'%epoch))
            torchvision.utils.save_image(vis.data,session.file('vis','%d.png'%epoch));
            
        torchvision.utils.save_image(vis.data,session.file('tmp.png'));
    '''
    
