

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

def minimize_affix(affix):
    m={}
    affix_min=[]
    for x in affix:
        if not x in m:
            m[x]=len(m)
        
        affix_min.append(m[x])
    
    return tuple(affix_min)

def permute_comp(comp,perm):
    comp_out=[]
    for row in comp:
        comp_out.append(tuple(minimize_affix([row[x] for x in perm])))
    
    comp_out=tuple(comp_out)
    return comp_out



def normalize_n_v2(comp):
    perms=list(itertools.permutations(range(len(comp[0]))))
    #Enumerate all perms of comp
    #Greedily minimize each perm by row
    
    comp_remap=[permute_comp(comp,perm) for perm in perms]
    m=sorted(comp_remap)[0]
    #m=tuple(zip(*m))
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



import math

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
        
        valid_cfgs.append(c)
    
    valid_cfgs=[(prefix,suffix,cfg) for cfg in valid_cfgs]
    #Try removing duplicates if any?
    
    return valid_cfgs

def validate_comps_paired_suffix(prefix,suffix,k=2):
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


#Generate all valid order K reduce operations with N dimensions
#Filter operations by dependency graph (NxN), (i,j)=1 => j is per-i and therefore can only be reduced before i
def generate_comps(N,K):
    affixes=calculate_affix_combos(K,K) #all compositions
    if N==1:
        comps=[[affix] for affix in affixes]
    else:
        comps=generate_comps(N-1,K)
        comps=list(itertools.product(comps,affixes))
        comps=[tuple(list(comp[0])+[comp[1]]) for comp in comps]
    
    comps_norm=[];
    for i,comp in enumerate(comps):
        if i%1000==0:
            print('N=%d normalizing comps %d/%d'%(N,i,len(comps)),end='\r')
        
        comps_norm.append(normalize_n_v2(comp))
    
    comps_norm=list(set(comps_norm));
    print('N=%d total %d comps'%(N,len(comps_norm)))
    return comps_norm

#Write a function that recursively generate all unique but possibly breakable comps
#Then write another function to remove breakable comps


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

