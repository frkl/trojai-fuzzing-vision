# Einsum Pooling: nD permutation symmetric neural nets from the ground up

Here we list a few applications that s

## III Use cases

### III.1 Learning matrix operations

Not the most efficient, but flexible.

### III.1 Learning embeddings

### Few-shot learning

### III.2 Knowledge graph completion

Use patterns in known relations to infer unknown relations


### III.3 Learning a Sudoku solver

### III.4 A few ARC-AGI problems



$X_{ab}$-type symmetry for matrix inverse


### III.2 $X_{aa}$-type symmetry for knowledge graph completion


### ARC-AGI
sudoku-type (abc)
4cd1b7b2.json

point cloud (abc)
00576224.json
ca8de6ea

section fill (aac)
a406ac07.json

abc
4f537728
physics(abc)
45bbe264
1a2e2828 (invariant)


Knowledge graph completion is another problem class where nD permutation symmetry is helpful. 

A knowledge graph stores knowledge as a collection of (entity_1, relation, entity_2) tuples, e.g. "" can be represented as a tuple (x,x,x). It is a natural yet powerful way of storing knowledge. 

A knowledge graph with $N_E$ entities and $N_R$ relations can be represented as a $N_E \times N_E \times N_R$ tensor $G(e1,e2,r)$, where $G(e1,e2,r)=1$ indicates that (e1,r,e2) is in the knowledge graph and $G(e1,e2,r)=0$ indicates that (e1,r,e2) is not in the knowledge graph.

Knowledge graph completion -- also known as link prediction -- is the task of inferring missing edges (or links) in a knowledge graph. For example, 
Simple as it seems, complex rules get exponentially harder to find and overfitting happens. It tests the ability of learning to reason has been one of the "AI complete" problems.

ARC-AGI introduction

Use one example. In this setup, we want to learn the "rules of the game" 


Process an ARC-AGI example into a knowledge graph.  
(Image of an example into a knowledge graph form)
```python
#00576224.json
data_train=[{"input": [[8, 6], [6, 4]], "output": [[8, 6, 8, 6, 8, 6], [6, 4, 6, 4, 6, 4], [6, 8, 6, 8, 6, 8], [4, 6, 4, 6, 4, 6], [8, 6, 8, 6, 8, 6], [6, 4, 6, 4, 6, 4]]}, {"input": [[7, 9], [4, 3]], "output": [[7, 9, 7, 9, 7, 9], [4, 3, 4, 3, 4, 3], [9, 7, 9, 7, 9, 7], [3, 4, 3, 4, 3, 4], [7, 9, 7, 9, 7, 9], [4, 3, 4, 3, 4, 3]]}]
data_test=[{"input": [[3, 2], [7, 8]], "output": [[3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8], [2, 3, 2, 3, 2, 3], [8, 7, 8, 7, 8, 7], [3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8]]}]

data_test=[{"input": [[0, 0, 0, 2], [0, 2, 2, 0], [2, 0, 0, 2], [0, 2, 2, 0], [2, 0, 2, 2], [0, 0, 0, 2], [7, 7, 7, 7], [6, 6, 0, 6], [6, 6, 6, 0], [0, 0, 0, 0], [6, 6, 0, 6], [6, 0, 6, 0], [0, 0, 6, 6]], "output": [[0, 0, 8, 0], [0, 0, 0, 8], [0, 8, 8, 0], [0, 0, 0, 0], [0, 8, 0, 0], [8, 8, 0, 0]]}]

data_train=[{"input": [[0, 2, 2, 0], [2, 0, 0, 0], [0, 2, 0, 2], [2, 2, 2, 2], [0, 0, 2, 0], [0, 0, 2, 2], [7, 7, 7, 7], [0, 6, 6, 0], [0, 0, 0, 0], [6, 6, 6, 6], [6, 6, 0, 6], [0, 6, 6, 6], [0, 0, 6, 0]], "output": [[8, 0, 0, 8], [0, 8, 8, 8], [0, 0, 0, 0], [0, 0, 0, 0], [8, 0, 0, 0], [8, 8, 0, 0]]}, {"input": [[2, 2, 0, 2], [2, 0, 2, 2], [2, 2, 0, 0], [0, 2, 0, 2], [0, 2, 2, 0], [2, 0, 0, 2], [7, 7, 7, 7], [6, 0, 6, 6], [0, 6, 0, 0], [0, 0, 0, 0], [0, 0, 0, 6], [6, 6, 0, 0], [6, 0, 6, 0]], "output": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 8, 8], [8, 0, 8, 0], [0, 0, 0, 8], [0, 8, 0, 0]]}, {"input": [[0, 0, 0, 2], [2, 0, 0, 0], [0, 2, 2, 2], [0, 0, 0, 2], [2, 0, 2, 0], [0, 2, 2, 0], [7, 7, 7, 7], [6, 0, 6, 6], [6, 0, 0, 6], [0, 6, 6, 6], [6, 0, 0, 0], [6, 0, 0, 6], [0, 0, 6, 0]], "output": [[0, 8, 0, 0], [0, 8, 8, 0], [8, 0, 0, 0], [0, 8, 8, 0], [0, 8, 0, 0], [8, 0, 0, 8]]}, {"input": [[2, 2, 0, 0], [0, 2, 2, 0], [2, 2, 0, 0], [2, 0, 0, 0], [0, 0, 0, 2], [2, 2, 0, 0], [7, 7, 7, 7], [6, 6, 6, 6], [6, 0, 6, 6], [6, 6, 0, 0], [0, 0, 0, 0], [6, 6, 0, 0], [0, 0, 6, 0]], "output": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 8, 8], [0, 8, 8, 8], [0, 0, 8, 0], [0, 0, 0, 8]]}]


sz=[max(torch.Tensor(x['input']).shape) for x in data_train]
sz+=[max(torch.Tensor(x['output']).shape) for x in data_train]
sz+=[max(torch.Tensor(x['input']).shape) for x in data_test]
sz+=[max(torch.Tensor(x['output']).shape) for x in data_test]
sz=max(sz)


#Describe KG into grid
def describe(input,output,ex_name='example'):
    links=[]
    links_test=[]
    H,W=len(input),len(input[0])
    for j in range(H):
        for k in range(W):
            links.append(['%s_input_%d_%d'%(ex_name,j,k),'input_of',ex_name])
            #links.append(['%s_input_%d_%d'%(ex_name,j,k),'has_tile_x','num_%d'%(j)])
            #links.append(['%s_input_%d_%d'%(ex_name,j,k),'has_tile_y','num_%d'%(k)])
            #links.append(['%s_input_%d_%d'%(ex_name,j,k),'has_tile_-x','num_%d'%(H-1-j)])
            #links.append(['%s_input_%d_%d'%(ex_name,j,k),'has_tile_-y','num_%d'%(W-1-k)])
            links.append(['%s_input_%d_%d'%(ex_name,j,k),'has_color','color_%d'%(input[j][k])])
    
    for j in range(H):
        for k in range(W):
            if j<H-1:
                links.append(['%s_input_%d_%d'%(ex_name,j,k),'next_x','%s_input_%d_%d'%(ex_name,j+1,k)])
            if k<W-1:
                links.append(['%s_input_%d_%d'%(ex_name,j,k),'next_y','%s_input_%d_%d'%(ex_name,j,k+1)])
    
    H,W=len(output),len(output[0])
    for j in range(H):
        for k in range(W):
            links.append(['%s_output_%d_%d'%(ex_name,j,k),'output_of',ex_name])
            #links.append(['%s_output_%d_%d'%(ex_name,j,k),'has_tile_x','num_%d'%(j)])
            #links.append(['%s_output_%d_%d'%(ex_name,j,k),'has_tile_y','num_%d'%(k)])
            #links.append(['%s_output_%d_%d'%(ex_name,j,k),'has_tile_-x','num_%d'%(H-1-j)])
            #links.append(['%s_output_%d_%d'%(ex_name,j,k),'has_tile_-y','num_%d'%(W-1-k)])
            links_test.append(['%s_output_%d_%d'%(ex_name,j,k),'has_color','color_%d'%(output[j][k])])
    
    for j in range(H):
        for k in range(W):
            if j<H-1:
                links.append(['%s_output_%d_%d'%(ex_name,j,k),'next_x','%s_output_%d_%d'%(ex_name,j+1,k)])
            if k<W-1:
                links.append(['%s_output_%d_%d'%(ex_name,j,k),'next_y','%s_output_%d_%d'%(ex_name,j,k+1)])
    
    return links,links_test
```
(Image illustrating how train/test split works)
Split data into train/test splits
```python
#Divide links into train_input, train_output, test_output
links_train_input=[]
links_train_output=[]
links_test_input=[]
links_test_output=[]

basic_links=[]
#for i in range(sz-1):
#    basic_links.append(['num_%d'%(i),'less_than','num_%d'%(i+1)])

#basic_links.append(['color_0','is_bg','color_0'])


all_links=basic_links
for i,ex in enumerate(data_train):
    links_i,links_test_i=describe(ex['input'],ex['output'],'train_%d'%i)
    links_train_input.append(basic_links+links_i)
    links_train_output.append(links_test_i)
    all_links=all_links+links_i+links_test_i

for i,ex in enumerate(data_test):
    links_i,links_test_i=describe(ex['input'],ex['output'],'test_%d'%i)
    links_test_input.append(basic_links+links_i)
    links_test_output.append(links_test_i)
    all_links=all_links+links_i+links_test_i

#Create entity/relation dictionaries
entities=sorted(list(set([x[0] for x in all_links]+[x[2] for x in all_links])))
relations=sorted(list(set([x[1] for x in all_links])))

def tuple_to_tensor(links,entities,relations):
    data=torch.LongTensor([[entities.index(x[0]),entities.index(x[2]),relations.index(x[1])] for x in links])
    X=torch.sparse_coo_tensor(data.t(),[1.0 for i in data],[len(entities),len(entities),len(relations)])
    return X.coalesce().to_dense()

def tuple_to_tensor2(links,links_test):
    all_links=links+links_test
    entities=sorted(list(set([x[0] for x in all_links]+[x[2] for x in all_links])))
    relations=sorted(list(set([x[1] for x in all_links])))
    print(len(entities),len(relations))
    return tuple_to_tensor(links,entities,relations).cuda(),tuple_to_tensor(links_test,entities,relations).cuda(),

X_train=[]
Y_train=[]
for i in range(len(links_train_input)):
    x,y=tuple_to_tensor2(links_train_input[i],links_train_output[i])
    X_train.append(x)
    Y_train.append(y)

X_test=[]
Y_test=[]
for i in range(len(links_test_input)):
    x,y=tuple_to_tensor2(links_test_input[i],links_test_output[i])
    X_test.append(x)
    Y_test.append(y)

```

Pooling and network design
```python
#Implements the following pooling terms: ['ZaaY->ZaaY','ZabY->ZabY','ZabY->ZbaY', 'ZabY->ZacY', 'ZabY->ZcbY', 'ZabY->ZcdY','ZabY,ZbcY->ZacY']
class einpool_aa(nn.Module):
    Kin=8
    Kout=7
    def forward(self,x):
        N,M,KH=x.shape[-3:]
        H=KH//8
        x=x.split(H,dim=-1)
        y0=x[0]
        y1=x[1].transpose(-2,-3)
        y2=x[2].sum(-2,keepdim=True).repeat(1,M,1)
        y3=x[3].sum(-3,keepdim=True).repeat(N,1,1)
        y4=x[4].sum([-2,-3],keepdim=True).repeat(N,M,1)
        y5=x[5].diagonal(dim1=-2,dim2=-3).diag_embed(dim1=-2,dim2=-3)
        y6=torch.einsum('abH,bcH->acH',x[6],x[7])
        y=torch.cat((y0,y1,y2,y3,y4,y5,y6),dim=-1)
        return y


net=einnet(ninput=len(relations),nh0=8,nh=32,noutput=len(relations),nstacks=6,pool=einpool_aa()).cuda()
```

Start training
```python

def forward(X,Y):
    ind=Y.nonzero()
    s=net(X)
    s1=s[ind[:,0],:,ind[:,2]]
    loss=F.cross_entropy(s1,ind[:,1])
    s2=s[:,ind[:,1],ind[:,2]+len(relations)].contiguous().t()
    loss=loss+F.cross_entropy(s2,ind[:,0])
    _,pred=s1.max(dim=-1)
    acc=pred.eq(ind[:,1]).float().mean()
    return loss,pred,acc


def split(X):
    mask=torch.rand_like(X).lt(0.8).to(X.dtype)
    X0=X.data*mask
    X1=X.data*(1-mask)
    return X0,X1


opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=0)

loss=[]
for i in range(1000000):
    net.zero_grad()
    for j in range(len(X_train)):
        loss_j,_,_=forward(X_train[j],Y_train[j])
        loss_j.backward()
        loss.append(float(loss_j.data))
    
    
    #for j in range(len(X_train)):
    #    X,Y=split(X_train[j]+Y_train[j])
    #    loss_j,_,_=forward(X,Y)
    #    loss_j.backward()
    #    loss.append(float(loss_j.data))
    
    
    opt.step()
    loss_i=sum(loss)/len(loss)
    print('iter %d, loss %f   '%(i,loss_i),end='\r')
    
    if i%1000==0:
        with torch.no_grad():
            for j in range(len(X_train)):
                loss_eval,pred,acc=forward(X_train[j],Y_train[j])
                print('iter %d, loss %f, loss_tr %f, acc %f,   '%(i,loss_i,loss_eval,acc))
                print(pred.view(-1).tolist())
            
            for j in range(len(X_test)):
                loss_eval,pred,acc=forward(X_test[j],Y_test[j])
                print('iter %d, loss %f, loss_eval %f, acc %f,   '%(i,loss_i,loss_eval,acc))
                print(pred.view(-1).tolist())
            
            loss=[]







```


## Final words

There have you, we've designed and verified an nD permutation symmetric network from the basic principles. The same bottom-up approach is applicable to many other types of symmetries, such as translation, scale, rotation. 
Interestingly, different types of symmetry lead to different rate of parameter sharing, and permutation symmetry is one that leads to a greater rate of parameter reduction. 


Designing general equivariant / invariant networks has been a hot topic. In fact, prior works [Max welling] discussed how to parameterize a linear layer given arbitrary linear transformations. ....  
Many network designs for specific types of nD permutation symmetry have also been proposed previously, such as . The use cases in our post are inspired by the [] work and a comment in its reviews. 
We'd like to encourage curious readers to read those works as well.


 

## Related readings

### Other neural net designs for nD permutation symmetry

### Designing neural nets with symmetry

### Designing neural nets using einsum pooling


