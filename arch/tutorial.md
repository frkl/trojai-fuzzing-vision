# Einsum Pooling: nD permutation symmetric neural nets from the ground up

## 
Classifying matrices? Processing a list of logits or embeddings? Predicting missing edges in graphs? For problems that have permutation symmetry along multiple axes, Einsum pooling networks (EinNet) could be a simple yet powerful tool in your arsenal.

(1 paragraph intro of Einsum) For example, an einsum operation implementing matrix multiplication with itself.

The basic building block of an EinNet consists of
1) an input MLP along the non-symmetric embedding dimension,
2) a normalization layer that clips the activation values for stability, e.g. tanh, softmax,
3) an einsum pooling layer that performs a set of specific einsum operation accoding to the desired symmetry to pool the input, and
4) an output MLP along the non-symmetric embedding dimension.

Obviously, if the einsum operations were picked to follow the desired symmetry, then permuting the input to the block would result in an output permuted in the same way, achieving permutation symmetry. Multiple equivariant blocks can be stacked to increase representation power. 

In this post, we'll walk through 1) a first principle derivation of the design of the network starting from the Taylor series of nD permutation symmetric functions requiring little math, and 2) from scratch implementations on several toy problems, including matrix inverse and knowledge graph completion using an ARC-AGI challenge case as an example.

## From the ground up

Our general approach here is to 



As a general rule of thumb, enforcing symmetry on a neural network induces parameter sharing.


### The intuition

Let's start from a simple 1-D permutation invariance case. Let's say we want to parameterize a function  
```math
y=f\left( \begin{bmatrix}x_{0} & x_{1} & x_{2} & x_{3}\end{bmatrix} \right)
```
to be invariant to permutation. Consider the Taylor series
```math
\begin{aligned}
f\left(\begin{bmatrix}x_{0} & x_{1} & x_{2} & x_{3}\end{bmatrix}\right)
= & c^{(0)} + 
\begin{bmatrix} c^{(1)}_{0} & c^{(1)}_{1} & c^{(1)}_{2} & c^{(1)}_{3}\end{bmatrix} 
\begin{bmatrix} x_{0} \\ x_{1} \\ x_{2} \\ x_{3}\end{bmatrix} \\ 
& +
\begin{bmatrix} x_{0} & x_{1} & x_{2} & x_{3}\end{bmatrix}
\begin{bmatrix} 
    c^{(2)}_{00} & c^{(2)}_{01} & c^{(2)}_{02} & c^{(2)}_{03} \\
    c^{(2)}_{10} & c^{(2)}_{11} & c^{(2)}_{12} & c^{(2)}_{13} \\
    c^{(2)}_{20} & c^{(2)}_{21} & c^{(2)}_{22} & c^{(2)}_{23} \\
    c^{(2)}_{30} & c^{(2)}_{31} & c^{(2)}_{32} & c^{(2)}_{33} 
\end{bmatrix} 
\begin{bmatrix} x_{0} \\ x_{1} \\ x_{2} \\ x_{3}\end{bmatrix}
+ \ldots
\end{aligned}
```

Since we want $f(\cdot)$ to be invariant to any permutation $P$, by definition we have 
```math
f\left(\begin{bmatrix}x_{0} \\ x_{1} \\ x_{2} \\ x_{3}\end{bmatrix}\right)-f\left(P\begin{bmatrix}x_{0} \\ x_{1} \\ x_{2} \\ x_{3}\end{bmatrix}\right)=0
```
That is 
```math
\begin{aligned}
& c^{(0)} - c^{(0)} + 
\left(\begin{bmatrix} c^{(1)}_{0} & c^{(1)}_{1} & c^{(1)}_{2} & c^{(1)}_{3}\end{bmatrix}
-\begin{bmatrix} c^{(1)}_{0} & c^{(1)}_{1} & c^{(1)}_{2} & c^{(1)}_{3}\end{bmatrix}P \right)
\begin{bmatrix} x_{0} \\ x_{1} \\ x_{2} \\ x_{3}\end{bmatrix} \\
\end{aligned}
```

```math 
\begin{aligned}
+
\begin{bmatrix} x_{0} & x_{1} & x_{2} & x_{3}\end{bmatrix}
\left(
\begin{bmatrix} 
    c^{(2)}_{00} & c^{(2)}_{01} & c^{(2)}_{02} & c^{(2)}_{03} \\
    c^{(2)}_{10} & c^{(2)}_{11} & c^{(2)}_{12} & c^{(2)}_{13} \\
    c^{(2)}_{20} & c^{(2)}_{21} & c^{(2)}_{22} & c^{(2)}_{23} \\
    c^{(2)}_{30} & c^{(2)}_{31} & c^{(2)}_{32} & c^{(2)}_{33} 
\end{bmatrix} 
-P^{T}
\begin{bmatrix} 
    c^{(2)}_{00} & c^{(2)}_{01} & c^{(2)}_{02} & c^{(2)}_{03} \\
    c^{(2)}_{10} & c^{(2)}_{11} & c^{(2)}_{12} & c^{(2)}_{13} \\
    c^{(2)}_{20} & c^{(2)}_{21} & c^{(2)}_{22} & c^{(2)}_{23} \\
    c^{(2)}_{30} & c^{(2)}_{31} & c^{(2)}_{32} & c^{(2)}_{33} 
\end{bmatrix} P
\right)
\begin{bmatrix} x_{0} \\ x_{1} \\ x_{2} \\ x_{3}\end{bmatrix}
+ \ldots =0
\end{aligned}
```

For this to be true for any $x$, the first order and second order coefficients need to independently satisfy for any permutation $P$, that

```math
\begin{aligned}
\begin{bmatrix} c^{(1)}_{0} & c^{(1)}_{1} & c^{(1)}_{2} & c^{(1)}_{3}\end{bmatrix}
-\begin{bmatrix} c^{(1)}_{0} & c^{(1)}_{1} & c^{(1)}_{2} & c^{(1)}_{3}\end{bmatrix}P 
=0
\end{aligned}
```
and 

```math
\begin{aligned}
\begin{bmatrix} 
    c^{(2)}_{00} & c^{(2)}_{01} & c^{(2)}_{02} & c^{(2)}_{03} \\
    c^{(2)}_{10} & c^{(2)}_{11} & c^{(2)}_{12} & c^{(2)}_{13} \\
    c^{(2)}_{20} & c^{(2)}_{21} & c^{(2)}_{22} & c^{(2)}_{23} \\
    c^{(2)}_{30} & c^{(2)}_{31} & c^{(2)}_{32} & c^{(2)}_{33} 
\end{bmatrix} 
-P^{T}
\begin{bmatrix} 
    c^{(2)}_{00} & c^{(2)}_{01} & c^{(2)}_{02} & c^{(2)}_{03} \\
    c^{(2)}_{10} & c^{(2)}_{11} & c^{(2)}_{12} & c^{(2)}_{13} \\
    c^{(2)}_{20} & c^{(2)}_{21} & c^{(2)}_{22} & c^{(2)}_{23} \\
    c^{(2)}_{30} & c^{(2)}_{31} & c^{(2)}_{32} & c^{(2)}_{33} 
\end{bmatrix} P
=0
\end{aligned}
```

Here for every $P$ we have an equation about coefficients $c$, and across all $P$ we have a set of equations in the form of $A\overrightarrow{c}=0$. Finding the null space of $A$ would give us the degrees of freedom that the coefficients $c$ can have. That's the key idea behind https://proceedings.mlr.press/v139/finzi21a/finzi21a.pdf and interested readers can read further.

For our specific case, in the first-order term, obviously we have
```math
c^{(1)}_{0}=c^{(1)}_{1}=c^{(1)}_{2}=c^{(1)}_{3}\triangleq b
```
In other words, the first order term only has 1 degree of freedom. The second order term turned out to have 2 degrees of freedom 

```math
\begin{aligned}
c^{(2)}_{ii}\triangleq c \\
c^{(2)}_{ij}\triangleq d, i\ne j
\end{aligned}
```

So our permutation invariant function turned out to look like
```math
\begin{aligned}
f\left(\begin{bmatrix}x_{0} & x_{1} & x_{2} & x_{3}\end{bmatrix}\right)
= & \textcolor{red}{a}+ 
\begin{bmatrix} \textcolor{orange}{b} & \textcolor{orange}{b} & \textcolor{orange}{b} & \textcolor{orange}{b}\end{bmatrix} 
\begin{bmatrix} x_{0} \\ x_{1} \\ x_{2} \\ x_{3}\end{bmatrix} \\ 
& +
\begin{bmatrix} x_{0} & x_{1} & x_{2} & x_{3}\end{bmatrix}
\begin{bmatrix} 
    \textcolor{blue}{c} & \textcolor{green}{d} & \textcolor{green}{d} & \textcolor{green}{d} \\
    \textcolor{green}{d} & \textcolor{blue}{c} & \textcolor{green}{d} & \textcolor{green}{d} \\
    \textcolor{green}{d} & \textcolor{green}{d} & \textcolor{blue}{c} & \textcolor{green}{d} \\
    \textcolor{green}{d} & \textcolor{green}{d} & \textcolor{green}{d} & \textcolor{blue}{c} 
\end{bmatrix} 
\begin{bmatrix} x_{0} \\ x_{1} \\ x_{2} \\ x_{3}\end{bmatrix}
+ \ldots
\end{aligned}
```



 




## Use cases

### $X_{ab}$-type symmetry for matrix inverse

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

#Implements the following pooling terms: 'ZabY->ZabY', 'ZabY->ZacY', 'ZacY->ZbcY', 'ZacY->ZbdY', 'ZacY,ZadY,ZbcY->ZbdY', 'ZacY,ZadY,ZaeY,ZbcY,ZbdY->ZbeY', 'ZadY,ZaeY,ZbdY,ZbeY,ZcdY->ZceY'
class einpool_ab(nn.Module):
	Kin=17
	Kout=7
	def forward(self,x):
		N,M,KH=x.shape[-3:]
		H=KH//17
		x=x.split(H,dim=-1)
		y0=x[0]
		y1=x[1].sum(-2,keepdim=True).repeat(1,M,1)
		y2=x[2].sum(-3,keepdim=True).repeat(N,1,1)
		y3=x[3].sum([-2,-3],keepdim=True).repeat(N,M,1)
		y4=torch.einsum('acH,adY,bcH->bdH',x[4],x[5],x[6])
		y5=torch.einsum('acH,bcH,adH,bdH,aeH->beH',x[7],x[8],x[9],x[10],x[11])
		y6=torch.einsum('adH,aeH,bdH,beH,cdH->ceH',x[12],x[13],x[14],x[15],x[16])
		y=torch.cat((y0,y1,y2,y3,y4,y5,y6),dim=-1)
		return y

#2-layer mlp
def mlp2(ninput,nh,noutput):
	return nn.Sequential(nn.Linear(ninput,nh),nn.GELU(),nn.Linear(nh,noutput))

class einnet(nn.Module):
	def __init__(self,ninput,nh0,nh,noutput,nstacks,pool):
		super().__init__()
		assert nstacks>=2
		self.t=nn.ModuleList()
		self.t.append(mlp2(ninput,nh,nh0*pool.Kin))
		for i in range(nstacks-2):
			self.t.append(mlp2(nh0*pool.Kout,nh,nh0*pool.Kin))
		
		self.t.append(mlp2(nh0*pool.Kout,nh,noutput))
		self.pool=pool
	
	def forward(self,x):
		h=self.t[0](x)
		for i in range(1,len(self.t)):
			hi=F.softmax(h.view(*h.shape[:-3],-1,h.shape[-1]),dim=-2).view(*h.shape)
			hi=self.t[i](self.pool(hi))
			if i<len(self.t)-1:
				h=h+hi
			else:
				h=hi
		
		return h

```


### $X_{aa}$-type symmetry for knowledge graph completion

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

#Describe KG into grid
def describe(input,output,ex_name='example'):
    links=[]
    links_test=[]
    H,W=len(input),len(input[0])
    for j in range(H):
        for k in range(W):
            links.append(['%s_input_%d_%d'%(ex_name,j,k),'input_of',ex_name])
            links.append(['%s_input_%d_%d'%(ex_name,j,k),'has_tile_x','num_%d'%(j)])
            links.append(['%s_input_%d_%d'%(ex_name,j,k),'has_tile_y','num_%d'%(k)])
            links.append(['%s_input_%d_%d'%(ex_name,j,k),'has_tile_-x','num_%d'%(H-1-j)])
            links.append(['%s_input_%d_%d'%(ex_name,j,k),'has_tile_-y','num_%d'%(W-1-k)])
            links.append(['%s_input_%d_%d'%(ex_name,j,k),'has_color','color_%d'%(input[j][k])])
    
    H,W=len(output),len(output[0])
    for j in range(H):
        for k in range(W):
            links.append(['%s_output_%d_%d'%(ex_name,j,k),'output_of',ex_name])
            links.append(['%s_output_%d_%d'%(ex_name,j,k),'has_tile_x','num_%d'%(j)])
            links.append(['%s_output_%d_%d'%(ex_name,j,k),'has_tile_y','num_%d'%(k)])
            links.append(['%s_output_%d_%d'%(ex_name,j,k),'has_tile_-x','num_%d'%(H-1-j)])
            links.append(['%s_output_%d_%d'%(ex_name,j,k),'has_tile_-y','num_%d'%(W-1-k)])
            links_test.append(['%s_output_%d_%d'%(ex_name,j,k),'has_color','color_%d'%(output[j][k])])
    
    return links,links_test
```
(Image illustrating how train/test split works)
Split data into train/test splits
```python
#Divide links into train_input, train_output, test_output
links_seen=[]
links_train_output=[]
links_test=[]
for i in range(5):
	links_seen.append(['num_%d'%(i),'less_than','num_%d'%(i+1)])

for i,ex in enumerate(data_train):
	links_i,links_test_i=describe(ex['input'],ex['output'],'train_%d'%i)
	links_seen+=links_i
	links_seen+=links_test_i
	links_train_output.append(links_test_i)

for i,ex in enumerate(data_test):
	links_i,links_test_i=describe(ex['input'],ex['output'],'test_%d'%i)
	links_seen+=links_i
	links_test+=links_test_i

#Create entity/relation dictionaries
all_links=links_seen+links_test
entities=sorted(list(set([x[0] for x in all_links]+[x[2] for x in all_links])))
relations=sorted(list(set([x[1] for x in all_links])))

def tuple_to_tensor(links,entities,relations):
	data=torch.LongTensor([[entities.index(x[0]),entities.index(x[2]),relations.index(x[1])] for x in links])
	X=torch.sparse_coo_tensor(data.t(),[1.0 for i in data],[len(entities),len(entities),len(relations)])
	return X.coalesce().to_dense()

X=tuple_to_tensor(links_seen,entities,relations).cuda()
Y_train=[tuple_to_tensor(links,entities,relations).cuda() for links in links_train_output]
Y_test=tuple_to_tensor(links_test,entities,relations).cuda()

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


net=einnet(ninput=len(relations),nh0=32,nh=128,noutput=len(relations),nstacks=6,pool=einpool_aa()).cuda()
```

Start training
```python

def forward(X,Y):
	ind=Y.nonzero()
	s=net(X)
	s=s[ind[:,0],:,ind[:,2]]
	loss=F.cross_entropy(s,ind[:,1])
	_,pred=s.max(dim=-1)
	acc=pred.eq(ind[:,1]).float().mean()
	return loss,pred,acc

def powerset(x):
	if len(x)==1:
		return [x]
	
	s=powerset(x[:-1])
	return s+[[x[-1]]]+[v+[x[-1]] for v in s]

splits=powerset(list(range(len(Y_train))))
import random

def divide(X):
	missing=torch.rand_like(X).gt(0.90).to(X.dtype)
	X_=X.data*(1-missing)
	Y_=X.data*missing
	return X_,Y_


opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=0)

loss=[]
for i in range(100000):
	net.zero_grad()
	for j in range(3):
		ind=splits[j%len(splits)]
		Y=torch.stack([Y_train[k] for k in ind],dim=0).sum(dim=0)
		#loss_j,_,_=forward(X-Y,Y)
		X_,Y_=divide(X-Y)
		loss_j,_,_=forward(X_,Y)
		loss_j.backward()
		loss.append(float(loss_j.data))
	
	opt.step()
	loss_i=sum(loss)/len(loss)
	print('iter %d, loss %f   '%(i,loss_i),end='\r')
	
	if i%1000==0:
		with torch.no_grad():
			for j in range(len(Y_train)):
				loss_eval,pred,acc=forward(X-Y_train[j],Y_train[j])
				print('iter %d, loss %f, loss_tr %f, acc %f,   '%(i,loss_i,loss_eval,acc))
				print(pred.view(-1).tolist())
			
			loss_eval,pred,acc=forward(X,Y_test)
			print('iter %d, loss %f, loss_eval %f, acc %f,   '%(i,loss_i,loss_eval,acc))
			print(pred.view(-1).tolist())
			loss=[]


with torch.no_grad():
	loss_eval,pred,acc=forward(X-Y_train[1]-Y_train[0],Y_train[1])
	print('iter %d, loss %f, loss_eval %f, acc %f,   '%(i,loss,loss_eval,acc))
	print(pred.view(-1).tolist())




```


## Final words

There have you, we've designed and verified an nD permutation symmetric network from the basic principles. The same bottom-up approach is applicable to many other types of symmetries, such as translation, scale, rotation. 
Interestingly, different types of symmetry lead to different rate of parameter sharing, and permutation symmetry is one that leads to a greater rate of parameter reduction. 

Another important. Our finding echoes an observation made by a recent work, that symmetric networks, while having fewer parameters, may not reduce compute. 


Designing general equivariant / invariant networks has been a hot topic. In fact, prior works [Max welling] discussed how to parameterize a linear layer given arbitrary linear transformations. ....  
Many network designs for specific types of nD permutation symmetry have also been proposed previously, such as . The use cases in our post are inspired by the [] work and a comment in its reviews. 
We'd like to encourage curious readers to read those works as well.






Although simple as it seems, 

## Related readings

### Other neural net designs for nD permutation symmetry

### Designing neural nets with symmetry

### Designing neural nets using einsum pooling


