import helper_r13 as helper
import torch
import os

m=os.path.join(helper.root(),'models','id-%08d'%3)
m=helper.engine(m)

data=m.load_examples()

for i in range(5,6):
    #loss,y=m.eval_grad(data[i])
    loss=m.eval(data[i])
    print(loss)

#im=data[0]['image'].unsqueeze(dim=0)


#outputs = m.model(im.cuda())


