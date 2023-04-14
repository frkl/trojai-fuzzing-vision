

import torch
import torch.nn as nn

import itertools
import string
import numpy



def powerset(n):
    s = list(range(n))
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


def calculate_affix_combos(N,K):
    if K==1:
        return [(0,)]
    
    comps_prev=calculate_affix_combos(N,K-1)
    comps=[];
    for c in comps_prev:
        n=min(max(list(c))+2,N)
        for i in range(n):
            comps.append(tuple(list(c)+[i]))
    
    return comps

#Check whether a comp is a multiplication of lower tier comps
def breakable(prefix,suffix):
    #generate all splits
    n=len(prefix)
    partitions=list(powerset(n))[1:-1]
    
    #check whether the split was perfect
    for p in partitions:
        p2=set(list(range(n))).difference(set(list(p)))
        p2=list(p2)
        
        prefix0=[prefix[i] for i in p]
        prefix1=[prefix[i] for i in p2]
        suffix0=[suffix[i] for i in p]
        suffix1=[suffix[i] for i in p2]
        
        prefix_shared=set(prefix0).intersection(set(prefix1))
        suffix_shared=set(suffix0).intersection(set(suffix1))
        
        if len(prefix_shared)==0 and len(suffix_shared)==0:
            return True
    
    return False

#Check whether a comp is a multiplication of lower tier comps
def breakable_n(comp):
    #generate all splits
    n=len(comp[0])
    k=len(comp)
    partitions=list(powerset(n))[1:-1]
    
    #check whether the split was perfect
    for ind in partitions:
        ind2=set(list(range(n))).difference(set(list(ind)))
        ind2=list(ind2)
        
        affix_0=[[x[i] for i in ind] for x in comp]
        affix_1=[[x[i] for i in ind2] for x in comp]
        
        no_shared=[len(set(affix_0[i]).intersection(set(affix_1[i])))==0 for i in range(len(comp))]
        
        if all(no_shared):
            return True
    
    return False

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

def remap(affix):
    #list all alternative numbering schemes
    v=set(list(affix))
    n=len(v)
    remaps=[]
    for p in itertools.permutations(range(n)):
        m=dict(zip(v,p))
        vp=tuple([m[x] for x in affix])
        remaps.append(vp)
    
    return remaps
    

def normalize(prefix,suffix):
    prefix_remap=remap(prefix)
    suffix_remap=remap(suffix)
    
    m=[tuple(sorted(list(zip(prefix,suffix)))) for prefix,suffix in itertools.product(prefix_remap,suffix_remap)]
    m=sorted(m)[0]
    m=list(zip(*m))
    return m[0],m[1]

def normalize_n(comp):
    comp_remap=[remap(x) for x in comp]
    
    m=[tuple(sorted(list(zip(*comp)))) for comp in itertools.product(*comp_remap)]
    m=sorted(m)[0]
    m=tuple(zip(*m))
    return m


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



def einstr(prefix_,suffix_):
    prefix_dict=string.ascii_lowercase
    suffix_dict=string.ascii_uppercase
    
    prefix=eliminate_singles(prefix_)
    suffix=eliminate_singles(suffix_)
    
    prepro=[]
    
    n=len(prefix)
    einstr=''
    einstr0=''
    arr_type=[]
    for i in range(n):
        einstr_i='Z'
        t=0
        if prefix[i]>=0:
            einstr_i+=prefix_dict[prefix[i]]
            t+=2
        
        if suffix[i]>=0:
            einstr_i+=suffix_dict[suffix[i]]
            t+=1
        
        einstr_j='Z'+prefix_dict[prefix_[i]]+suffix_dict[suffix_[i]]
        
        arr_type.append(t)
        prepro.append((einstr_j+'->'+einstr_i,(0,)));
        einstr+=einstr_i
        einstr0+=einstr_j
        if i<n-1:
            einstr+=','
            einstr0+=','
    
    einstr+='->Z'
    einstr0+='->Z'
    return einstr0,einstr,arr_type,prepro

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
        t=0
        s='Z'
        for i,x in enumerate(comp):
            if x==-1:
                t+=(n-1-i)**2
            else:
                s+=vocab[(i,x)]
        
        return s,t
    
    
    #Regroup by item
    comp_0=list(zip(*comp)) #Keep a copy of original to produce
    comp=list(zip(*[eliminate_singles(affix) for affix in comp]))
    
    prepro=[]
    
    n=len(comp)
    einstr=''
    arr_type=[]
    for i in range(n):
        einstr_0,_=translate(comp_0[i],vocab)
        einstr_i,t=translate(comp[i],vocab)
        
        
        
        arr_type.append(t)
        prepro.append((einstr_0+'->'+einstr_i,(0,)));
        einstr+=einstr_i
        einstr0+=einstr_i
        if i<n-1:
            einstr+=','
    
    einstr+='->Z'
    return einstr,arr_type,prepro


#einstr_n(((0, 0, 1, 1, 2), (0, 1, 0, 1, 0), (0, 1, 0, 1, 1)))



def parse_line(line):
    return [x for x in line.split('\n')[-1].split(' ') if not x=='']

def parse_einpath(path):
    p=path[0][1:]
    rows=path[1].split('\n')
    flops=float(parse_line(rows[4])[-1])
    memory=float(parse_line(rows[6])[-2])
    cmd=[parse_line(row)[1] for row in rows[10:]]
    return flops,memory,list(zip(cmd,p))



def einopt(prefix,suffix,w=1000,h=1000):
    n=len(prefix)
    s0,s,t,prepro=einstr(prefix,suffix)
    A=[numpy.ones((1,)),numpy.ones((1,h)),numpy.ones((1,w)),numpy.ones((1,w,h))]
    
    path=numpy.einsum_path(s,*[A[ti] for ti in t],optimize='optimal')
    flops,memory,cmd=parse_einpath(path)
    #print(path[0])
    #print(path[1])
    return s0,s,flops,memory,prepro+cmd



def einopt_n(comp,sz):
    s0,s,t,prepro=einstr(comp)
    
    #generate example matrices
    #for 
    torch.ones(sz)
    
    A=[numpy.ones((1,)),numpy.ones((1,h)),numpy.ones((1,w)),numpy.ones((1,w,h))]
    
    path=numpy.einsum_path(s,*[A[ti] for ti in t],optimize='optimal')
    flops,memory,cmd=parse_einpath(path)
    #print(path[0])
    #print(path[1])
    return s0,s,flops,memory,prepro+cmd


def parse_einstr(s):
    i=s.find('->')
    s_var=s[:i]
    s_out=s[i+2:]
    s_ops=s_var.split(',')
    return s_ops,s_out

import math
def ein_slice_x(s_ops,x,*ops,sz=1000):
    #to split or not
    for i,s in enumerate(s_ops):
        if x in s:
            n=ops[i].shape[s.index(x)]
            break
    
    if n<=sz:
        return [ops]
    
    nslices=math.ceil(n/sz)
    
    #generate slices
    ops_=[];
    for i,op in enumerate(ops):
        if x in s_ops[i]:
            dim=s_ops[i].index(x)
            ops_.append(op.split(sz,dim=dim))
        else:
            ops_.append([op for j in range(nslices)])
    
    ops_=[[op[i] for op in ops_] for i in range(nslices)];
    return ops_


def ein_slice(s,*ops,sz=1000,one=False):
    s_ops,s_out=parse_einstr(s)
    ''.join(s_ops)
    literals=set(''.join(s_ops))
    slices=[ops]
    for x in literals:
        if not x in s_out and (one or ''.join(s_ops).count(x)>1):
            slices=list(itertools.chain(*[ein_slice_x(s_ops,x,*ops_,sz=sz) for ops_ in slices]))
    
    return slices


#out=ein_slice('zij,zij->z',torch.Tensor(1,5000,3000),torch.Tensor(1,5000,3000),sz=1000)



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
            
            if sz is None:
                args_i=[args_i]
            else:
                args_i=ein_slice(cmd[0],*args_i,sz=sz,one=True)
            
            #print(cmd, [x.shape for x in args_i],[x.shape for x in args])
            #print(cmd[0],sz,[x.shape for x in args_i[0]])
            y=[torch.einsum(cmd[0],*[x.to(device) for x in arg]) for arg in args_i]
            y=torch.stack(y,dim=0).sum(dim=0)
            args.append(y);
        
        return y

class perm_inv2(nn.Module):
    def __init__(self,comps,w=1000,h=1000):
        super().__init__()
        self.nets=nn.ModuleList();
        self.comps=comps
        
        mem=[]
        compute=[]
        einstrs=[]
        
        for prefix,suffix in comps:
            path=einopt(prefix,suffix,w=w,h=h)
            mem_i=path[-2]/1e9
            compute_i=path[-3]/1e12
            if compute_i>20:
                continue;
            
            
            m=einnet(path[-1])
            m.comp=(prefix,suffix)
            
            self.nets.append(m)
            einstrs.append(path[0])
            mem.append(mem_i)
            compute.append(compute_i)
        
        self.einstrs=einstrs
        print('compute: %f TF'%(sum(compute)))
        print('memory: %f G'%(max(mem)))
    
    def forward(self,x,sz=None,device=None):
        sz_=list(x.shape)
        w_=sz_[-1]
        h_=sz_[-2]
        x=x.view(-1,sz_[-2],sz_[-1])
        hs=[];
        for i,net in enumerate(self.nets):
            n=len(net.comp[0])
            nprefix=len(set(self.comps[i][0]))
            nsuffix=len(set(self.comps[i][1]))
            ops=[x]*n
            if sz is None:
                h=net(*ops)
            else:
                ops_slices=ein_slice(self.einstrs[i],*ops,sz=sz)
                h=[net(*ops,sz=sz,device=device) for ops in ops_slices]
                
                h=torch.stack(h,dim=0).sum(dim=0)
                #print('%s %d chunks'%(self.einstrs[i],len(ops_slices)),[list(op.shape) for op in ops_slices[0]],end='\r')
            
            h=h/(h_**nprefix)/(w_**nsuffix)
            
            hs.append(h)
        
        hs=torch.stack(hs,dim=-1)
        #hs=hs.view(-1,len(self.nets))
        hs=hs.view(*(sz_[:-2]+[len(self.nets)]))
        return hs


class perm_inv2_paired_suffix(nn.Module):
    def __init__(self,comps,w=1000,h=1000):
        super().__init__()
        self.nets=nn.ModuleList();
        
        mem=[]
        compute=[]
        einstrs=[]
        for prefix,suffix,cfg in comps:
            path=einopt(prefix,suffix,w=w,h=h)
            mem_i=path[-2]/1e9
            compute_i=path[-3]/1e12
            if compute_i>20:
                continue;
            
            
            m=einnet(path[-1])
            m.comp=(prefix,suffix,cfg)
            self.nets.append(m)
            einstrs.append(path[0])
            mem.append(mem_i)
            compute.append(compute_i)
        
        self.einstrs=einstrs
        print('compute: %f TF'%(sum(compute)))
        print('memory: %f G'%(max(mem)))
    
    def forward(self,xs,sz=None,device=None):
        sz_=[list(x.shape) for x in xs]
        xs=[x.view(-1,x.shape[-2],x.shape[-1]) for x in xs]
        
        hs=[];
        for i,net in enumerate(self.nets):
            cfg=(net.comp[-1])
            ops=[xs[j] for j in cfg]
            if sz is None:
                h=net(*ops)
            else:
                ops_slices=ein_slice(self.einstrs[i],*ops,sz=sz)
                #print('%s %d chunks'%(self.einstrs[i],len(ops_slices)),end='\r')
                h=[net(*ops,sz=sz,device=device) for ops in ops_slices]
                
                h=torch.stack(h,dim=0).sum(dim=0)
                #print('%s %d chunks'%(self.einstrs[i],len(ops_slices)),[list(op.shape) for op in ops_slices[0]],end='\r')
            
            hs.append(h)
        
        hs=torch.stack(hs,dim=-1)
        hs=hs.view(*(sz_[0][:-2]+[len(self.nets)]))
        return hs

def validate_comps_paired(prefix,suffix,k=2):
    #Both need to participate
    #Prefix needs to perfectly split
    #Solution might not exist
    n=len(prefix)
    cfgs=list(itertools.product(*([list(range(k))]*n)))
    valid_cfgs=[];
    for c in cfgs:
        #Check whether all items are present
        if not len(set(c))==k:
            continue;
        
        #Check whether prefix split perfectly
        m={}
        valid=True
        for i in range(len(prefix)):
            if not prefix[i] in m:
                m[prefix[i]]=c[i]
            elif m[prefix[i]]!=c[i]:
                valid=False
                break;
        
        if not valid:
            continue;
        
        valid_cfgs.append(c)
    
    valid_cfgs=[(prefix,suffix,cfg) for cfg in valid_cfgs]
    #Try removing duplicates if any?
    
    return valid_cfgs


    

def generate_comps(n=5):
    prefix=calculate_affix_combos(n,n)
    suffix=calculate_affix_combos(n,n)
    
    comps=list(itertools.product(prefix,suffix))
    #print(len(comps))
    
    comps_n=list(set([normalize(p,s) for p,s in comps]))
    comps_=[(p,s) for p,s in comps_n if not breakable(p,s)]
    
    
    comps_=list(set(comps_))
    return comps_

def generate_comps_n(n=5,k=3):
    comps=[calculate_affix_combos(n,n) for i in range(k)]
    comps=list(itertools.product(*comps))
    comps_n=[];
    for i,comp in enumerate(comps):
        if i%1000==0:
            print('Normalizing comps %d/%d'%(i,len(comps)),end='\r')
        
        comps_n.append(normalize_n(comp))
    
    print('');
    comps_n=list(set(comps_n))
    print('%d remains after normalization'%(len(comps_n)))
    comps_=[comp for comp in comps_n if not breakable_n(comp)]
    comps_=list(set(comps_))
    print('%d final'%(len(comps_)))
    return comps_



'''

comps=[];
for i in range(1,6):
    print(i)
    comps+=generate_comps(i)

net=perm_inv2(comps)
net(torch.Tensor(3,15,10))

comps_paired=[]
for prefix,suffix in comps:
    comps_paired+=validate_comps_paired(prefix,suffix,2)

net=perm_inv2_paired_suffix(comps_paired)
net((torch.Tensor(3,15,10),torch.Tensor(3,8,10)))




paths=[einopt(*comp) for comp in comps]
[p for p in paths if p[1]>1e9]


print(len(comps_))

path=einopt(*comps_[0])
tmp=net(path[-1])
x=torch.ones(1,100,120)
y=tmp(x,x,x,x,x)

[x for x in path[1].split('\n')[-1].split(' ') if not x=='']

'''

