import torch
#import util.perm_inv_v2 as inv
import util.perm_inv_v1b as inv

comps=inv.generate_comps_n(3,5)
ds=[(2,1),(2,0)]
comps2=[comp for comp in comps if inv.dependency(comp,ds)]
comps3=[comp for comp in comps2 if len(set(comp[-1]))<=3]
print(len(comps3))



# 1        100  10      300   100  5
#ntriggers nim ntrials nboxes classid 5

#results=inv.einopt_n(comps3[0],(100,10,300,100,5))
#print(results)


net=inv.perm_inv_n(comps3,(50,5,150,50,6))

x=torch.zeros(1,100,10,300,100,6,device='cuda')
with torch.no_grad():
    y=net(x)

print(y.shape)

#print(results)
a=0/0


s=inv.einstr_n(((0, 0, 1, 1, 2), (0, 1, 0, 1, 0), (0, 1, 0, 1, 1)))
print(s)
