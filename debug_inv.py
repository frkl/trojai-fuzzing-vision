import torch
#import util.perm_inv_v2 as inv
import util.perm_inv_v1b as inv
import numpy


comps=inv.generate_comps_n(3,5)
ds=[(2,1),(2,0)]
comps2=[comp for comp in comps if inv.dependency(comp,ds)]
comps3=[comp for comp in comps2 if len(set(comp[-1]))<=3]
print(len(comps3))

#Rough estimate of naive implementation flops and memory given einsum string
def einsum_flops(s,shapes):
    #Infer dimensionality for each literal
    #FLOPS: all literals multiplied together
    #Memory: max size of all terms
    lhs,rhs=s.split('->')
    lhs=lhs.replace(' ','').split(',')
    terms=lhs+[rhs]
    sz={}
    for i in range(len(terms)-1):
        for j,c in enumerate(terms[i]):
            if not c in sz:
                sz[c]=shapes[i][j]
    
    flops=numpy.prod([sz[c] for c in sz])
    memory=max([numpy.prod([sz[c] for c in x]) for x in terms])
    return flops,memory

# einsum_flops('ij,jk->ik',[[10,100],[100,30],[10,30]])


#Compute all viable einsum paths between two variables
def invariant_op2()


1,2,3,4,5
6,7,8,9,10



# 1        100  10      300   100  5
#ntriggers nim ntrials nboxes classid 5

#results=inv.einopt_n(comps3[0],(100,10,300,100,5))
#print(results)


def discrete_einsum()

net=inv.perm_inv_n(comps3,(50,5,150,50,6))

x=torch.zeros(1,100,10,300,100,6,device='cuda')
with torch.no_grad():
    y=net(x)

print(y.shape)

#print(results)
a=0/0


s=inv.einstr_n(((0, 0, 1, 1, 2), (0, 1, 0, 1, 0), (0, 1, 0, 1, 1)))
print(s)



