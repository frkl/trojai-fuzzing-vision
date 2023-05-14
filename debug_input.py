import helper_r13_v2 as helper
import util.db as db
import torch.nn.functional as F
import torch
import copy
import torchvision
import torch.optim as optim

interface=helper.get(2)
ex_clean=interface.load_examples()
ex_clean2=interface.more_clean_examples()
ex_poisoned=interface.load_poisoned_examples()
ex_poisoned2=interface.target_to_source(ex_poisoned)
ex_poisoned3=interface.replace(ex_poisoned)


def loss_and_grad_input(interface,example):
    ex=copy.deepcopy(example)
    interface.model.zero_grad()
    ex['image']=ex['image'].cuda().requires_grad_()
    loss=interface.eval_loss(ex)
    loss.backward()
    
    return {'loss':float(loss),'grad':ex['image'].grad.data.clone().cpu()}


def opt(interface,example):
    ex=copy.deepcopy(example)
    ex['image']=ex['image'].cuda().requires_grad_()
    opt=optim.Adam([ex['image']],lr=1e-2);
    loss0=interface.eval_loss(ex)
    for i in range(5):
        interface.model.zero_grad()
        loss=interface.eval_loss(ex)
        loss.backward()
        opt.step()
    
    loss_after=interface.eval_loss(ex)
    
    return {'loss':float(loss0),'loss_after':float(loss_after),'grad':ex['image'].grad.data.clone().cpu(),'image':ex['image'].data.clone().cpu()}

#data_clean=[loss_and_grad_input(interface,ex_clean[i]) for i in range(len(ex_clean))];
#data_clean2=[loss_and_grad_input(interface,ex_clean2[i]) for i in range(len(ex_clean2))];
data_poisoned=[loss_and_grad_input(interface,ex_poisoned[i]) for i in range(len(ex_poisoned))];
data_poisoned3=[opt(interface,ex_poisoned3[i]) for i in range(len(ex_poisoned3))];

#data_poisoned2=[opt(interface,ex_poisoned2[i]) for i in range(len(ex_poisoned2))];
[x['loss'] for x in data_poisoned3]
[x['loss_after'] for x in data_poisoned3]




im0=torch.cat([ex_poisoned3[i]['image'] for i in range(len(ex_poisoned3))],dim=-1).squeeze(dim=0)
im=torch.cat([x['image'].clamp(min=0,max=1) for x in data_poisoned3],dim=-1).squeeze(dim=0).abs()
#im=im/im.max()
im=torch.cat((im0,im),dim=-2)
torchvision.utils.save_image(im,'im.png')

#Compute layer-wise correlation
layer_names=[name for name,param in interface.model.named_parameters()]

def grad_norm(data):
    grads=[];
    for layer in layer_names:
        g=[];
        for x in data:
            for v in x['grad']:
                if v['name']==layer:
                    g.append(v['grad'].view(-1))
                    break
        
        g=torch.stack(g,dim=0)
        grads.append({'layer':layer,'grad':g})
    
    m=[{'layer':g['layer'],'m':g['grad'].norm(dim=-1)} for g in grads]
    
    return db.Table.from_rows(m)



m_clean=grad_norm(data_clean)
m_clean2=grad_norm(data_clean2)
m_poisoned=grad_norm(data_poisoned)
m_poisoned2=grad_norm(data_poisoned2)

fnames=[x['fname'] for x in list(ex_clean.rows())+ex_clean2+list(ex_poisoned.rows())+list(ex_poisoned2.rows())]
data=torch.cat((m_clean['m'],m_clean2['m'],m_poisoned['m'],m_poisoned2['m']),dim=-1)


def write_csv(fname,data,cols,rows):
    f=open(fname,'w')
    f.write(',')
    for x in cols:
        f.write('%s,'%x)
    
    f.write('\n')
    
    for i,x in enumerate(rows):
        f.write('%s,'%x)
        for j in range(len(cols)):
            f.write('%f,'%data[i][j])
        
        f.write('\n')
    
    f.close()

write_csv('debug.csv',data,fnames,layer_names)
a=0/0

'''
grads_clean=[ [for i in range(layers)]  ]



data=data_clean+data_clean2+data_poisoned+data_poisoned2




#Compute layer-wise correlation
layer_names=[name for name,param in interface.model.named_parameters()]

correlations=[]
for k,name in enumerate(layer_names):
    print('%s'%name,end='\r')
    S=torch.zeros(len(data),len(data))
    for i in range(len(data)):
        for j in range(len(data)):
            assert data[i]['grad'][k]['name']==name and  data[j]['grad'][k]['name']==name
            w0=F.normalize(data[i]['grad'][k]['grad'].cuda().view(-1),dim=-1)
            w1=F.normalize(data[j]['grad'][k]['grad'].cuda().view(-1),dim=-1)
            S[i,j]=float((w0*w1).sum())
    
    correlations.append({'name':name,'S':S})


torch.save(correlations,'debug.pt')


layer=1
(correlations[layer]['S'][:5,-5:]*1000).long()
(correlations[layer]['S'][:5,:5]*1000).long()


grads=[ [x['grad'] for x in data[i]['grad']] for i in range(len(data))]

h1=[fv1(g) for g in grads]
h1=torch.stack(h1,dim=0)

h2=[]
for i,g1 in enumerate(grads):
    for j,g2 in enumerate(grads):
        print('%d,%d / %d'%(i,j,len(grads)))
        h2.append(fv2(g1,g2))

h2=torch.stack(h2,dim=0)
h2=h2.view(len(grads),len(grads),-1)

def fv1(g):
    v=torch.stack([x.cuda().mean() for x in g],dim=-1)
    v2=torch.stack([(x.cuda()**2).mean() for x in g],dim=-1)
    h=[]
    h.append(v.mean())
    h.append(v.mean()**2)
    h.append(v2.mean())
    h.append((v**2).mean())
    h=torch.stack(h,dim=-1)
    return h

def fv2(gx,gy):
    vx=torch.stack([x.cuda().mean() for x in gx],dim=-1)
    #vx2=torch.stack([(x.cuda()**2).mean() for x in gx],dim=-1)
    vy=torch.stack([x.cuda().mean() for x in gy],dim=-1)
    #vy2=torch.stack([(x.cuda()**2).mean() for x in gy],dim=-1)
    vxy=torch.stack([(gx[i].cuda()*gy[i].cuda()).mean() for i in range(len(gx))],dim=-1)
    
    h=[]
    
    h.append(vx.mean())
    h.append(vy.mean())
    h.append(vx2.mean())
    h.append(vy2.mean())
    h.append(vx.mean()**2)
    h.append(vy.mean()**2)
    h.append((vx**2).mean())
    h.append((vy**2).mean())
    
    
    h.append(vx.mean()*vy.mean())
    h.append((vx*vy).mean())
    h.append(vxy.mean())
    
    h=torch.stack(h,dim=-1)
    return h


def abstract(examples):
    N=len(examples)
    for 
'''
