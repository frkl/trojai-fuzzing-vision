# Einsum Pooling: nD permutation symmetric neural nets from the ground up


## II Engineering a Network with Permutation Symmetry

From matrices to sets to symbolic processing, permutation symmetry is found in many problems and requires extra attention during modeling. When handled properly however, permutation symmetry is also a blessing. As we have learned in the previous section, if parameterized properly, permutation symmetry has the potential to exponentially reduce parameter count and compute for highly efficient learning. At the other end of the spectrum, reciting the success recipe of deep learning, we can scale the latent dimension and stack equivariant layers to create exponentially more expressive networks at the same parameter count and compute as a regular network.

![Network with Equivariant Backbone](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png)


Devil's in the details, in this section we'll walk through the design of permutation symmetric neural networks for various types of permutation symmetry.


There are many places where you'll see permutation symmetry and they often come in different forms. So in Section II.1 we'll first start from a summary of common types of permutation symmetry. 
And then Section II.2 will discuss the design of permutation equivariant layers. Permutation symmetry turns out to be closely tied to tensor contractions. That would allow us to synthesize efficient high-order permutation equivariant layers automatically in a procedural manner.
And finally in Section II.3 we discuss further optimizations that helps practical implementation. 

### II.1 Common types of permutation symmetry

In the following table we list a few common problems with different types of permutation symmetry.

| Problem       | Illustration  | Symmetry type  | Dependency |
| ------------- |:-------------:|:--------------:|:---------- |
|               |               |                |            |

Multiple dimensions, joint permutation and dependency are common themes here. To aid discussions, we also use a custom notation to describe the specific type of permutation symmetry, to capture both the input shape and the unique permutation axes. A fully independent batch dimension Z and a non-symmetric latent dimension H may be added optionally.


### II.2 Creating a permutation equivariant layer with einsum pooling

Across all types of permutation symmetry, as we learned in Section I through Taylor series, it turns out that tensor contractions are are all you need for parameterizing permutation invariant and equivariant layers, which can then be stacked into a deep network.

**How to create an equivariant layer given permutation symmetry type?**
Our answer is two fully connected layers with a pooling layer in between. 

<details>

<summary>
Primer: Tensor contractions and the einsum notation
</summary> 

Intuitively, tensor contractions like
```math
Y_{ij}=\sum_k \sum_l X_{ik} X_{lk} X_{lj} 
```
create a new tensor that has the same shape as the input while summing over unused dimensions. They achieve a permutation equivariant effect. And tensor contractions like
```math
y=\sum_i \sum_j \sum_k \sum_l X_{ik} X_{lk} X_{lj} 
```
that sum over all dimensions achieve a permutation invariant effect. 

As the math equations can get quite long, we will use the [einsum notation](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) which represents a tensor contraction using the indices involved. It is widely used across deep learning frameworks to denote tensor contractions. For example,
```python
Y=einsum('Zik,Zlk,Zlj->Zij',X,X,X)
y=einsum('Zik,Zlk,Zlj->Z',X,X,X)
```
Here a batch dimension Z is added to make sure the right hand side is not empty.  

</details>  


Let us use a 1D + latent `aH`-type equivariant constraint as an example to illustrate the design.

The Taylor Series parameterization up to order 2 is
```python
Y_abH=einsum('a->ba',a_H)
     +einsum('ab,ca->cb',b0_HH,X_aH)
     +einsum('ab,ca->db',b1_HH,X_aH)
     +einsum('abc,da,db->dc',c0_HHH,X_aH,X_aH)
     +einsum('abc,da,db->ec',c1_HHH,X_aH,X_aH)
     +einsum('abc,da,eb->dc',c2_HHH,X_aH,X_aH)
     +einsum('abc,da,eb->fc',c3_HHH,X_aH,X_aH)
     +...
```

We can immediately see that the order-1 terms have $H^2$ parameters and order-2 terms have $H^3$ parameters, which would naturally need a low-rank($=K$) treatment, such as

```python
Y_abH=einsum('a->ba',a_H)
     +einsum('ka,kb,ca->cb',b0U_KH,b0V_KH,X_aH)
     +einsum('ka,kb,ca->db',b1U_KH,b1V_KH,X_aH)
     +einsum('ka,kb,kc,da,db->dc',c0U,c0V,c0W,X_aH,X_aH)
     +einsum('ka,kb,kc,da,db->ec',c1U,c1V,c1W,X_aH,X_aH)
     +einsum('ka,kb,kc,da,eb->dc',c2U,c2V,c2W,X_aH,X_aH)
     +einsum('ka,kb,kc,da,eb->fc',c3U,c3V,c3W,X_aH,X_aH)
     +...
```

We can move the order-0 parameters, as well as $U$, $V$, $W$ matrices into fully connected layers along $H$ that perform input preprocessing and output post_processing. So the end result is two linear layers with pooling in between, and for pooling we need 

```python
Y_abH_0V=einsum('ck->ck',X_aH_0U)
Y_abH_1V=einsum('ck->dk',X_aH_1U)
Y_abH_0W=einsum('dk,dk->dk',X_aH_0U,X_aH_0V)
Y_abH_1W=einsum('dk,dk->ek',X_aH_1U,X_aH_1V)
Y_abH_2W=einsum('dk,ek->dk',X_aH_2U,X_aH_2V)
Y_abH_3W=einsum('dk,ek->fk',X_aH_3U,X_aH_3V)
...
```

Notice that `'dk,ek->fk'` can be composed with `'ck->dk'` for each operand individually, and then combine using `'dk,dk->dk'`. As we can stack more layers, not all pooling operations are needed and less pooling operations would reduce network complexity. In fact, this might be a good point to step back and ask: **Given equivariance type, e.g. `aH`, how can we identify the minimum yet sufficient set of pooling operations?**

The following recipe might be helpful for designing pooling operations given equivariance type in practice:

1) Enumerate all valid and unique einsum operations up to order-k that are compatible with the given equivariance type. For example `einsum('ab,bc->ac',X_ab,X_ab)` is compatible with `aa`-type equivariance, but not compatible with `ab`-type equivariance. Also notice that `ba,ac->bc` is the just a renaming of `ab,bc->ac`. There is a graph homomorphism problem under the hood for listing unique einsum operations and interested readers can dig deeper.

2) Filter einsum operations based on dependency requirement of the given equivariance type. For example `einsum('ab,cb->cb',X_ab,X_ab)` satisfy `b->a` dependency but does not satisfy `a->b` dependency for `ab`-type equivariance.

3) Filter out order-2+ "breakable" operations that can be divided into two lower order terms with a simple pointwise multiplication. For example `ab,cb,cd,ad->ad` can be divided into `ab,cb,cd->ad` and `ad->ad` which can the be put together with `ab,ab->ab`, so it is not necessary as long as the lower order terms exist. 

4) Normalize the rotation of input/output terms. For example for `aa`-type equivariance, `ab,cb->ca` is not necessary because it can be achieved with `ab,bc->ac`, through applying rotations `ab->ba` on the input and output.

5) Remove order-2+ operations that expand new dimensions in the output term. For example `ab,bc->ad` is redundant because it can be achieved through `ab,bc->ac` followed by a dimension expansion operation `ab->ac`.

An algorithm that properly de-duplicates through compositions remains to be developed. But after all the filtering listed here, there is usually a quite compact initial set of pooling operations for further optimizations.

The following is a quick lookup table of pooling operations for a few common equivariance types.


| Symmetry type | Order | Pooling operation(s) |
| ------------- |-------|:------------------:|
| `aH`          | 1     | `aH->aH`, `aH->bH` |
|               | 2     | `aH,aH->aH`  |
|               | 3+    | No need  |
| `abH`         | 1     | `abH->abH`, `abH->cbH`, `abH->acH` |
|               | 2     | `abH,abH->abH` |
|               | 3     | `abH,cbH,cdH->adH` |
|               | 4     | No need |
|               | 5     | `abH,acH,dbH,dcH,deH->aeH`, `abH,acH,dbH,dcH,ecH->ebH` |
| `aaH`         | 1     | `abH->abH`,`aaH->aaH`,`abH->baH`,`abH->cbH` |
|               | 2     | `abH,bcH->acH`,`abH,abH->abH` |
|               | 3+    | No need  |


*Did you know: `'ab,cb,cd->ad'` which is linear self-attention is an order-3 term for `ab`-type equivariance. Self-attention operation by itself is equivariant to not only token permutation but also latent permutation, although other linear layers in the transformer architecture do not retain the latent symmetry.*


### II.3 Putting everything together: The Equivariant Einsum Network

The following diagram shows the final network design.

![Complete design](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png)

In addition to layer stacking of equivariant layesr, the following features are further added:

1) Softmax normalization before einsum pooling to prevent high order einsum from exploding.
2) GELU nonlinearity between equivariant layers to add to the depth and create bottlenecks.
3) Residual connections for better optimization dynamics

Another consideration in practice is [einsum path optimization](https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html). For example, the einsum string `ab,dc,ae,ac,db->de` by default is programmed to be computed pairwise from left to right. By the third term, a large factor `abcde` would be created and stress the memory. Instead, if we compute pairwise via path `ab,db->ad`, `ac,dc->ad`, `ad,ad->ad` and `ad,ae->de`, the largest intermediate factor would only be 2-dimensional and the computation can also be done much faster. For modeling complex higher-order interations under certain types of symmetries, large einsums may be unavoidable, and computing them might be an interesting compute challenge.

Putting it all together, here is a reference Pytorch implementation of an Equivariant Einsum Network that would served as the backbone, to be followed by averaging for dimensions that need invariance. 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

#Implements minimal aH-type pooling
class einpool_a(nn.Module):
    fan_in=4
    fan_out=3
    ndims=1
    def forward(self,x_):
        x=x_.view(-1,*x_.shape[-2:]) # Apply pooling only to the last 2 dims, supposedly `aH`
        N,KH=x.shape[-3:]
        H=KH//self.fan_in
        x=x.split(H,dim=-1)
        y0=x[0]
        y1=x[1].sum(-2,keepdim=True).repeat(1,N,1)
        y2=x[2]*x[3]
        y=torch.cat((y0,y1,y2),dim=-1)
        y=y.view(*x_.shape[:-1],-1) #Recover original tensor shape
        return y

#Implements minimal aaH-type pooling
class einpool_aa(nn.Module):
    fan_in=8
    fan_out=6
    ndims=2
    def forward(self,x_):
        x=x_.view(-1,*x_.shape[-3:]) # Apply pooling only to the last 3 dims, supposedly `aaH`
        N,M,KH=x.shape[-3:]
        H=KH//self.fan_in
        x=x.split(H,dim=-1)
        y0=x[0]
        y1=x[1].diagonal(dim1=-2,dim2=-3).diag_embed(dim1=-2,dim2=-3)
        y2=x[2].transpose(-2,-3)
        y3=x[3].sum(-2,keepdim=True).repeat(1,1,M,1)
        y4=x[4]*x[5]
        y5=torch.einsum('ZabH,ZbcH->ZacH',x[6],x[7])
        y=torch.cat((y0,y1,y2,y3,y4,y5),dim=-1)
        y=y.view(*x_.shape[:-1],-1) #Recover original tensor shape
        return y

#Implements order-3 abH-type pooling
class einpool_ab(nn.Module):
    fan_in=8
    fan_out=5
    ndims=2
    def forward(self,x_):
        x=x_.view(-1,*x_.shape[-3:]) # Apply pooling only to the last 3 dims, supposedly `abH`
        N,M,KH=x.shape[-3:]
        H=KH//self.fan_in
        x=x.split(H,dim=-1)
        y0=x[0]
        y1=x[1].sum(-2,keepdim=True).repeat(1,1,M,1)
        y2=x[2].sum(-3,keepdim=True).repeat(1,N,1,1)
        y3=x[3]*x[4]
        y4=torch.einsum('ZacH,ZbcH,ZadH->ZbdH',x[5],x[6],x[7])
        y=torch.cat((y0,y1,y2,y3,y4),dim=-1)
        y=y.view(*x_.shape[:-1],-1) #Recover original tensor shape
        return y

#2-layer mlp with GELU
def mlp2(ninput,nh,noutput):
    return nn.Sequential(nn.Linear(ninput,nh),nn.GELU(),nn.Linear(nh,noutput))

#Equivariant EinNet backbone
class einnet(nn.Module):
    #Instantiate the network
    #    ninput/noutput -- number of input/output dimensions
    #    nh0 -- pooling dimensions, like head_dim in transformers
    #    nh -- latent dimensions 
    #    nstacks -- number of einsum pooling stacks
    #    pool -- einsum pooling operation. Needs to provide fan_in, 
    #            fan_out factors, and ndims
    def __init__(self,ninput,nh0,nh,noutput,nstacks,pool):
        super().__init__()
        self.t=nn.ModuleList()
        self.t.append(mlp2(ninput,nh,nh0*pool.fan_in))
        for i in range(nstacks-1):
            self.t.append(mlp2(nh0*pool.fan_out,nh,nh0*pool.fan_in))
        
        self.t.append(mlp2(nh0*pool.fan_out,nh,noutput))
        self.pool=pool
    
    # Forward call
    #    x: tensor shape matches equivariance type, e.g. *abH
    def forward(self,x):
        h=self.t[0](x)
        for i in range(1,len(self.t)):
            hi=F.softmax(h.view(*h.shape[:-self.pool.ndims-1],-1,h.shape[-1]),dim=-2).view(*h.shape)
            hi=self.t[i](self.pool(hi))
            #Residual connection
            if i<len(self.t)-1:
                h=h+hi
            else:
                h=hi
        
        return h

#Example usage
#    net=einnet(1,16,64,1,2,einpool_ab())
```
