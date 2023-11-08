

import torch
import torch.nn as nn

import itertools
import string
import numpy

from functools import reduce
from itertools import chain

data=torch.load('0.pt')

#Generate a "sparse" matrix from bounding boxes
#model trigger examples boxes class clean/poison/gt [dims]
N=len(data['pred_clean'])
ntriggers=data['ntriggers']
nex=N//ntriggers

idx=[]
val=[]

for i in range(N):
    trigger_id=i//nex
    ex_id=i%nex
    #clean boxes
    boxes=data['pred_clean'][i]['boxes']/300
    labels=data['pred_clean'][i]['labels']
    scores=data['pred_clean'][i]['scores']
    for boxid in range(len(boxes)):
        c=int(labels[boxid])
        v=torch.cat((boxes[boxid].view(-1),scores[boxid].view(-1)),dim=-1);
        idx.append((trigger_id,ex_id,boxid,c,0))
        val.append(v.cpu())
    
    #poison boxes
    boxes=data['pred_poisoned'][i]['boxes']/300
    labels=data['pred_poisoned'][i]['labels']
    scores=data['pred_poisoned'][i]['scores']
    for boxid in range(len(boxes)):
        c=int(labels[boxid])
        v=torch.cat((boxes[boxid].view(-1),scores[boxid].view(-1)),dim=-1);
        idx.append((trigger_id,ex_id,boxid,c,1))
        val.append(v.cpu())
    
    #gt boxes
    boxes=data['pred_gt'][i]['boxes']/300
    labels=data['pred_gt'][i]['labels']
    scores=data['pred_gt'][i]['scores']
    for boxid in range(len(boxes)):
        c=int(labels[boxid])
        v=torch.cat((boxes[boxid].view(-1),scores[boxid].view(-1)),dim=-1);
        idx.append((trigger_id,ex_id,boxid,c,2))
        val.append(v.cpu())


sparse_data=torch.sparse_coo_tensor(torch.LongTensor(idx).t(),torch.stack(val,dim=0))
sparse_data=sparse_data.coalesce()

#For index shared between two operands
#First organize the operands by those indicies
#Because only we only need to work on the part that have the same shared indicies 
def map_shared_indicies(stuff):
    s,s_shared,idx,v=stuff
    ind=[s.index(x) for x in s_shared]
    idx_shared=tuple([idx[i] for i in ind])
    return {idx_shared:[(idx,v)]}

def reduce_shared_indicies(d1,d2):
    for k in d2:
        if k in d1:
            d1[k]+=d2[k]
        else:
            d1[k]=d2[k]
    
    return d1

#For two parts with the same shared indicies, perform einsum
def map_einsum(stuff):
    ind,d0,d1=stuff
    idx=[]
    val=[]
    for k0,v0 in d0:
        for k1,v1 in d1:
            k=k0+k1
            kout=[k[i] for i in ind]
            vout=v0*v1 #Straight up multiply
            idx.append(kout)
            val.append(vout)
    
    return idx,val

def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)

def sparse_einsum_filter(s,idx,val):
    s=list(s)
    #find chars in s that appeared more than once
    occ={}
    for i,ch in enumerate(s):
        if not ch in occ:
            occ[ch]=[]
        
        occ[ch].append(i)
    
    checklist={ch:occ[ch] for ch in occ if len(occ[ch])>1}
    keep=[]
    for i,ind in enumerate(idx):
        k=True
        for ch in checklist:
            if not all_equal([ind[j] for j in checklist[ch]]):
                k=False;
                break;
        
        if k:
            keep.append(i)
    
    idx=[idx[i] for i in keep]
    val=[val[i] for i in keep]
    return idx,val

def sparse_einsum_1(s0,sout,x0):
    t0=time.time()
    #handles einsums of sparse matx that does not touch the last dimension
    s0=list(s0)
    sout=list(sout)
    assert s0[-1]==sout[-1]
    #Create sout mapping
    #Figure out how to map input idx to output idx
    out_ind=[s0.index(x) for x in sout[:-1]]
    
    #Get index and value
    ind0=x0.indices().t().tolist()
    v0=x0.values()
    sz0=x0.size()
    
    ind0,v0=sparse_einsum_filter(s0,ind0,v0)
    print('filter %.2f'%(time.time()-t0))
    
    idx=[[ind[i] for i in out_ind] for ind in ind0]
    val=v0
    sz=[sz0[i] for i in out_ind]+[sz0[-1]]
    print('rename %.2f'%(time.time()-t0))
    
    if len(idx)>0:
        result=torch.sparse_coo_tensor(torch.LongTensor(idx).to(x0.device).t(),torch.stack(val,dim=0),size=sz)
        result=result.coalesce()
    else:
        result=torch.empty(sz, layout=torch.sparse_coo)
    
    result=result.coalesce()
    print('coalesce %.2f'%(time.time()-t0))
    return result


#naive einsum
#Divide inputs into pockets by 
#Find valid input/outputs

def sparse_einsum_2(s0,s1,sout,x0,x1):
    t0=time.time()
    from functools import reduce
    from itertools import chain
    #handles einsums of sparse matx that does not touch the last dimension
    s0=list(s0)
    s1=list(s1)
    sout=list(sout)
    s_shared=list(set(s0[:-1]).intersection(set(s1[:-1])))
    assert s0[-1]==s1[-1]
    assert s0[-1]==sout[-1]
    #Create sout mapping
    #Figure out how to map input two idx to output idx
    s=s0[:-1]+s1[:-1]
    out_ind=[s.index(x) for x in sout[:-1]]
    
    #Get index and value
    ind0=x0.indices().t().tolist()
    v0=x0.values()
    sz0=x0.size()
    ind1=x1.indices().t().tolist()
    v1=x1.values()
    sz1=x1.size()
    
    ind0,v0=sparse_einsum_filter(s0,ind0,v0)
    ind1,v1=sparse_einsum_filter(s1,ind1,v1)
    print('filter %.2f'%(time.time()-t0))
    
    d0_=map(map_shared_indicies,[(s0,s_shared,ind0[i],v0[i]) for i in range(len(ind0))])
    
    d0={}
    for d in d0_:
        for k in d:
            if not k in d0:
                d0[k]=d[k]
            else:
                d0[k]+=d[k]
    
    #d0=reduce(reduce_shared_indicies,d0_)
    
    print('d0 agg %.2f'%(time.time()-t0))
    d1_=map(map_shared_indicies,[(s1,s_shared,ind1[i],v1[i]) for i in range(len(ind1))])
    
    d1={}
    for d in d1_:
        for k in d:
            if not k in d1:
                d1[k]=d[k]
            else:
                d1[k]+=d[k]
    
    #d1=reduce(reduce_shared_indicies,d1_)
    print('d1 agg %.2f'%(time.time()-t0))
    
    k_shared=list(set(d0.keys()).intersection(set(d1.keys())))
    
    jobs=[(out_ind,d0[k],d1[k]) for k in k_shared]
    v=map(map_einsum,jobs)
    #Faster than reduce
    idx=[]
    val=[]
    for x,y in v:
        idx+=x
        val+=y
    
    sz=sz0[:-1]+sz1[:-1]
    sz=[sz[i] for i in out_ind]+[sz0[-1]]
    
    #idx,val=reduce(lambda x,y:(x[0]+y[0],x[1]+y[1]),v)
    print('mul %.2f'%(time.time()-t0))
    
    if len(idx)>0:
        result=torch.sparse_coo_tensor(torch.LongTensor(idx).to(x0.device).t(),torch.stack(val,dim=0),size=sz)
        result=result.coalesce()
    else:
        result=torch.empty(sz, layout=torch.sparse_coo)
    
    print('coalesce %.2f'%(time.time()-t0))
    return result

def sparse_einsum(s,*operands):
    i=s.find('->')
    lhs=s[:i]
    rhs=s[i+2:]
    
    lhs=lhs.split(',')
    lhs=[x.rstrip(' ').lstrip(' ') for x in lhs]
    rhs=rhs.rstrip(' ').lstrip(' ')
    
    sz_operands=[numpy.product(op.size()) for op in operands]
    if max(sz_operands)<1e7:
        y=torch.einsum(s,*[op.to_dense() for op in operands])
        if len(rhs)>1:
            return y.to_sparse(sparse_dim=len(rhs)-1)
        else:
            return y
    
    if len(lhs)==2:
        return sparse_einsum_2(lhs[0],lhs[1],rhs,*operands)
    elif len(lhs)==1:
        return sparse_einsum_1(lhs[0],rhs,*operands)
    else:
        a=0/0


import time
s0='aacdef'
s1='abghif'
sout='egf'
x0=sparse_data.clone()
x1=sparse_data.clone()

y0=sparse_einsum('abegkm->abekm',x0)
y1=sparse_einsum('abehlm->abelm',x0)
y2=sparse_einsum('aceikm->aekm',x0)
y3=sparse_einsum('adfjlm->alm',x0)


y5=sparse_einsum('aekm,abekm->abem',y2,y0)
y6=sparse_einsum('abem,abelm->alm',y5,y1)
y7=sparse_einsum('alm,alm->m',y6,y3)




'''

tmp=sparse_einsum_1('aacdef','acf',x0)
tmp2=torch.einsum('aacdef->acf',x0.to_dense()[:27,:27])

print(tmp.shape,tmp2.shape,x0.shape)

diff2=(tmp.to_dense()-tmp2)/(tmp2.abs()+1e-20)
print(diff2.max())
print(diff2.min())

result=sparse_einsum_2(s0,s1,sout,x0,x1)
with torch.no_grad():
    result2=torch.einsum(s0+','+s1+'->'+sout,x0.to_dense()[:27,:27],x1.to_dense())

diff=result.to_dense()-result2
diff2=(result.to_dense()-result2)/(result2.abs()+1e-20)
print(diff2.max())
print(diff2.min())
'''


def count(x):
    c={}
    for v in x:
        if not v in c:  
            c[v]=0
        
        c[v]=c[v]+1
    
    c2={}
    for v in c:
        if not c[v] in c2:
            c2[c[v]]=[v]
        else:
            c2[c[v]].append(v)
    
    return c2;

def eliminate_singles(affix):
    c=count(affix)
    if 1 in c:
        affix_=[]
        for x in affix:
            if x in c[1]:
                affix_.append(-1)
            else:
                affix_.append(x)
        
        return affix_
    else:
        return affix



def einstr_n(comp):
    #Create a lookup table of literals
    ch=string.ascii_lowercase+string.ascii_uppercase[:-1]
    n=0
    vocab={}
    for i,affix in enumerate(comp):
        for j in set(affix):
            assert(n<len(ch))
            vocab[(i,j)]=ch[n]
            n=n+1
    
    def translate(comp,vocab):
        n=len(comp)
        s=''
        for i,x in enumerate(comp):
            if x!=-1:
                s+=vocab[(i,x)]
        
        s+='Z'
        return s
    
    
    #Regroup by item
    comp_0=list(zip(*comp)) #Keep a copy of original to produce
    
    comp=list(zip(*([tuple(eliminate_singles(affix)) for affix in comp])))
    
    prepro=[]
    
    n=len(comp)
    einstr=''
    einstr0=''
    targets=[]
    for i in range(n):
        einstr_0=translate(comp_0[i],vocab)
        einstr_i=translate(comp[i],vocab)
        
        prepro.append((einstr_0+'->'+einstr_i,(0,)));
        einstr+=einstr_i
        einstr0+=einstr_0
        if i<n-1:
            einstr+=','
            einstr0+=','
    
    targets=sorted(set(targets))
    einstr+='->Z'+''.join(targets)
    einstr0+='->Z'+''.join(targets)
    noutputs=len(targets)
    return einstr0,einstr,prepro,noutputs


def parse_line(line):
    return [x for x in line.split('\n')[-1].split(' ') if not x=='']

def parse_einpath(path):
    p=path[0][1:]
    rows=path[1].split('\n')
    flops=float(parse_line(rows[4])[-1])
    memory=float(parse_line(rows[6])[-2])
    cmd=[parse_line(row)[1] for row in rows[10:]]
    return flops,memory,list(zip(cmd,p))


def einopt_n(comp,sz):
    #div=max(round(min(sz)/3),1)
    #sz=[max(i//div,1) for i in sz]
    
    s0,s,prepro,noutputs=einstr_n(comp)
    print(s0,s,prepro)
    A=torch.ones(1,*sz).cuda()
    #generate example matrices
    A=[torch.einsum(s[0],A) for s in prepro]
    A=[x.cpu().numpy() for x in A]
    
    path=numpy.einsum_path(s,*A,optimize='optimal')
    flops,memory,cmd=parse_einpath(path)
    #post process Z to last for every einpath
    
    cmd_=[]
    for x,p in cmd:
        i=x.find('->')
        lhs=x[:i]
        rhs=x[i+2:]
        
        lhs=lhs.split(',')
        lhs=[x.rstrip(' ').lstrip(' ') for x in lhs]
        rhs=rhs.rstrip(' ').lstrip(' ')
        
        lhs=[v.replace('Z','')+'Z' for v in lhs]
        rhs=rhs.replace('Z','')+'Z'
        x_=','.join(lhs)+'->'+rhs
        cmd_.append((x_,p))
    
    return {'s0':s0,'s':s,'flops':flops,'memory':memory,'cmd':prepro+cmd_,'noutputs':noutputs}

result=einopt_n(comp,(10,10,10,10,10))

def parse_einstr(s):
    i=s.find('->')
    s_var=s[:i]
    s_out=s[i+2:]
    s_ops=s_var.split(',')
    return s_ops,s_out

import math
class einnet(nn.Module):
    def __init__(self,cmd):
        super().__init__()
        self.cmd=cmd
    
    def forward(self,*args,sz=None,device=None):
        args=[x for x in args]
        for cmd in self.cmd:
            
            args_i=[args[i] for i in cmd[1][::-1]]
            for i in cmd[1][::-1]:
                args.pop(i)
            
            print(cmd[0],[x.shape for x in args_i])
            y=sparse_einsum(cmd[0],*[x.to(device) for x in args_i])
            args.append(y);
        
        return y

import torch
import util.perm_inv_v2 as inv2
import util.perm_inv_v1b as inv
import numpy


comps=inv.generate_comps_n(3,5)
#comps=inv2.generate_comps(5,4)
ds=[(3,0),(3,1),(3,2)]
comps2b=[comp for comp in comps if inv.dependency(comp,ds)]

nets=[]

for comp in comps2b:
    result=einopt_n(comp,(10,10,10,10,10))
    net=einnet(result['cmd'])
    nets.append(net)

x0=x0.cuda()
y=[]
for i,net in enumerate(nets):
    print('----- net %d/%d ---------'%(i,len(nets)))
    y.append(net(x0,x0,x0))

y=[]



class perm_inv_n(nn.Module):
    def __init__(self,comps,sz):
        super().__init__()
        self.nets=nn.ModuleList();
        self.comps=comps
        self.N=len(comps[0])
        
        mem=[]
        compute=[]
        einstrs=[]
        noutputs=[];
        
        for comp in comps:
            path=einopt_n(comp,sz)
            mem_i=path['memory']/1e9
            compute_i=path['flops']/1e12
            if compute_i>20:
                continue;
            
            
            m=einnet(path['cmd'])
            m.comp=comp
            
            self.nets.append(m)
            einstrs.append(path['s0'])
            noutputs.append(path['noutputs'])
            mem.append(mem_i)
            compute.append(compute_i)
        
        self.einstrs=einstrs
        print('remaining ops: %d'%(len(self.nets)))
        print('compute: %f TF'%(sum(compute)))
        print('memory: %f G'%(max(mem)))
        print('est. output dimensions: %d'%sum([sz[-1]**k for k in noutputs]))
    
    def forward(self,x):
        sz_=list(x.shape)
        x=x.view(-1,*sz_[-self.N:])
        hs=[];
        for i,net in enumerate(self.nets):
            n=len(net.comp[0])
            ops=[x]*n
            h=net(*ops)
            h=h.view(*(sz_[:-self.N]+[-1]))
            hs.append(h)
        
        hs=torch.cat(hs,dim=-1)
        #hs=hs.view(*(sz_[:-self.N]+[len(self.nets)]))
        return hs

