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

## I. In Theory: Paramterizing Symmetric Functions in Taylor Series

Speaking of permutation invariance, you may already have your faviorite ways to design invariant neural networks for certain types of problems. 
But here we'll introduce a simple yet general Taylor series-based technique necessary for studying complex equivariance patterns. Specifically, we aim to design efficient universal learners -- that can represent any such invariant or equivariant functions -- for linear symmetries -- invariance or equivariance to linear transformations on the input. 
For advanced readers, the general idea follows [Equivariant Multilayer Perceptrons (EMLP)](https://github.com/mfinzi/equivariant-MLP) with a high-order twist. 

The steps are: 
1) Express a function as a Taylor series, 
2) Write the symmetry constraint as equations about the coefficients,
3) Solve the equations for free parameters and the parameter sharing pattern.

Let's start from 1D permutation invariance as an example to demonstrate how the technique works.


### I.1 Example: 1D permutation invariance

Let's say we want to make a function 
```math
y=f\left( \begin{bmatrix}x_{0} & x_{1} & x_{2}\end{bmatrix} \right)
```
invariant to permutation of $x_0$, $x_1$, $x_2$.

Consider the Taylor series
```math
\begin{aligned}
f&\left(\begin{bmatrix}x_{0} & x_{1} & x_{2}\end{bmatrix}\right) \\
= & a + 
\begin{bmatrix} b_0 & b_1 & b_2\end{bmatrix} 
\begin{bmatrix} x_0 \\ x_1 \\ x_2 \end{bmatrix} +
\begin{bmatrix} x_0 & x_{1} & x_{2}\end{bmatrix}
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02}\\
    c_{10} & c_{11} & c_{12}\\
    c_{20} & c_{21} & c_{22}
\end{bmatrix} 
\begin{bmatrix} x_{0} \\ x_{1} \\ x_{2} \end{bmatrix}
+ \ldots
\end{aligned}
```

Since we want $f(\cdot)$ to be invariant to any permutation matrix $P$, the invariant constraint says 
```math
f\left(\begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix}\right)=f\left(
    \begin{bmatrix} 
      &   &  \\
      & P &  \\
      &   &  
    \end{bmatrix} 
    \begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix}\right)
```
For our Taylor series form, this means that all order-k coefficients need to match. That is for any permutation matrix $P$

```math
\begin{aligned}
a = & a \\
\begin{bmatrix} b_0 & b_1 & b_2\end{bmatrix}  = &
\begin{bmatrix} b_0 & b_1 & b_2\end{bmatrix} 
    \begin{bmatrix} 
      &   &  \\
      & P &  \\
      &   &  
    \end{bmatrix} 
\\
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02}\\
    c_{10} & c_{11} & c_{12}\\
    c_{20} & c_{21} & c_{22}
\end{bmatrix} 
=& 
\begin{bmatrix} 
    &   &  \\
    & P^T &  \\
    &   &  
\end{bmatrix} 
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02}\\
    c_{10} & c_{11} & c_{12}\\
    c_{20} & c_{21} & c_{22}
\end{bmatrix} 
    \begin{bmatrix} 
      &   &  \\
      & P &  \\
      &   &  
    \end{bmatrix} \\
& \ldots
\end{aligned}
```
Which are all linear equations about coefficients $a$, $b_i$ and $c_{ij}$. So we can just enumerate all $P$ to get all the equations, and then solve them. For $b_i$ for example, enumerating different permutations $P$ would give

```math
\begin{bmatrix} b_0 \\ b_1 \\ b_2\end{bmatrix}  = 
\begin{bmatrix} b_0 \\ b_2 \\ b_1\end{bmatrix}  = 
\begin{bmatrix} b_1 \\ b_0 \\ b_2\end{bmatrix}  = 
\begin{bmatrix} b_1 \\ b_2 \\ b_0\end{bmatrix}  = 
\begin{bmatrix} b_2 \\ b_0 \\ b_1\end{bmatrix}  = 
\begin{bmatrix} b_2 \\ b_1 \\ b_0\end{bmatrix} 
```

Which is more than enough to say $b_0=b_1=b_2$. So the order-1 term has only 1 degree of freedom.

For $c_i$ there are more equations, but it turns out that solving the equations across all permutations would yield 
```math
c_{00}=c_{11}=c_{22} \\
c_{01}=c_{10}=c_{10}=c_{12}=c_{20}=c_{21}
```
So the order 2 term has 2 degrees of freedom, one for the diagonal and one for the off-diagonal.

Apply what we have learned, we can now write

```math
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{0} & x_{1} & x_{2}\end{bmatrix} \right)\\
= & a + 
\begin{bmatrix} b & b & b\end{bmatrix} 
\begin{bmatrix} x_0 \\ x_1 \\ x_2 \end{bmatrix} +
\begin{bmatrix} x_0 & x_{1} & x_{2}\end{bmatrix}
\begin{bmatrix} 
    c_{0} & c_{1} & c_{1}\\
    c_{1} & c_{0} & c_{1}\\
    c_{1} & c_{1} & c_{0}
\end{bmatrix} 
\begin{bmatrix} x_{0} \\ x_{1} \\ x_{2} \end{bmatrix}
+ \ldots
\end{aligned}
```
For a total of 4 free parameters up to order 2, instead of 13 free parameters without the invariance constraint. More generally, for $N$ inputs, we still only need 4 parameters instead of $N^2+N+1$ parameters. So parameterizing with symmetry can sometimes **reduce parameter count** exponentially. 

We can further simplify by focusing on the free parameters
```math
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{0} & x_{1} & x_{2}\end{bmatrix} \right)\\
= & a + b \sum_i x_i + (c_0-c_1) \sum_i x_i^2 + c_1 \sum_{i} \sum_{j} x_i x_j + \ldots \\
= & a + b \sum_i x_i + (c_0-c_1) \sum_i x_i^2 + c_1 (\sum_{i}x_i )^2 + \ldots
\end{aligned}
```
An important effect of this simplification is **reduced compute**. It now requires $O(N)$ compute for $N$ inputs instead of $O(N^2)$ for order-2. 

In math terms, the number of free parameters is the dimensionality of the null space of the symmetry constraints. The degrees of freedoms can be numerically calculated from the basis of this null space which is one of the many innovations in [1]. But note that as the basis is often not unique, numerical solvers tend to return linear combinations instead of simple terms, which makes it difficult to simplify. So there is still some fun in manual analysis.

### I.2 Exercises
If you are interested in going a little deeper, test yourself on a list of exercises for new insights.

**1D translation.** Parameterize function 
```math
y=f\left( \begin{bmatrix}x_{0} & x_{1} & x_{2} & x_{3}\end{bmatrix} \right) =f\left( \begin{bmatrix} x_{3} & x_{0} & x_{1} & x_{2}\end{bmatrix} \right)
```

<details>

<summary> 
Solution
</summary>


According to equivariant constraints, the coefficients of the Taylor series satisfy
```math
\begin{aligned}
a = & a \\
\begin{bmatrix} b_0 & b_1 & b_2 & b_3\end{bmatrix}  = &
\begin{bmatrix} b_1 & b_2 & b_3 & b_0\end{bmatrix} 
\\
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02} & c_{03}\\
    c_{10} & c_{11} & c_{12} & c_{13}\\
    c_{20} & c_{21} & c_{22} & c_{23}\\
    c_{30} & c_{31} & c_{32} & c_{33}
\end{bmatrix} 
=& 
\begin{bmatrix} 
    c_{11} & c_{12} & c_{13} & c_{10} \\
    c_{21} & c_{22} & c_{23} & c_{20} \\
    c_{31} & c_{32} & c_{33} & c_{30} \\
    c_{01} & c_{02} & c_{03} & c_{00} \\
\end{bmatrix} 
& \ldots
\end{aligned}
```
Which means there are 6 free parameters up to order-2.
```math
\begin{aligned}
b_0=b_1=&b_2=b_3 \\
c_{00}=c_{11}=&c_{22}=c_{33} \\
c_{01}=c_{12}=&c_{23}=c_{30} \\
c_{02}=c_{13}=&c_{20}=c_{31} \\
c_{03}=c_{10}=&c_{21}=c_{32} \\
\end{aligned}
```
Note: considering Hessian transpose symmetry, we would additionally have $c_{ij}=c_{ji}$ and reduce number of free parameters to 5. For now, let us assume that multiply among $x_i$ does not commute.

The parameterization with 6 parameters has an unrolled circular convolution on the order-2 term.

```math
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{0} & x_{1} & x_{2} & x_{3}\end{bmatrix} \right)\\
= & a + 
\begin{bmatrix} b & b & b &b\end{bmatrix} 
\begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \end{bmatrix} +
\begin{bmatrix} x_0 & x_{1} & x_{2} & x_3\end{bmatrix}
\begin{bmatrix} 
    c_{0} & c_{1} & c_{2} & c_{3}\\
    c_{3} & c_{0} & c_{1} & c_{2}\\
    c_{2} & c_{3} & c_{0} & c_{1}\\
    c_{1} & c_{2} & c_{3} & c_{0}
\end{bmatrix} 
\begin{bmatrix} x_{0} \\ x_{1} \\ x_{2} \\ x_{3} \end{bmatrix}
+ \ldots
\end{aligned}
```

Computing the 2nd order term naively would involve $N^2+N$ multiplies for length-$N$ input. A simplification like

```math
\begin{aligned}
&\begin{bmatrix} x_0 & x_{1} & x_{2} & x_3\end{bmatrix}
\begin{bmatrix} 
    c_{0} & c_{1} & c_{2} & c_{3}\\
    c_{3} & c_{0} & c_{1} & c_{2}\\
    c_{2} & c_{3} & c_{0} & c_{1}\\
    c_{1} & c_{2} & c_{3} & c_{0}
\end{bmatrix} 
\begin{bmatrix} x_{0} \\ x_{1} \\ x_{2} \\ x_{3} \end{bmatrix} \\

=& \frac{c_0+c_2}{2} (x_0+x_2)^2 + \frac{c_0-c_2}{2}(x_0-x_2)^2 \\ 
& + \frac{c_1+c_3}{2} (x_0+x_2)(x_1+x_3) + \frac{c_1-c_3}{2}(x_0-x_2)(x_1-x_3) \\ 
& + \frac{c_3+c_1}{2} (x_1+x_3)(x_0+x_2) + \frac{c_3-c_1}{2}(x_1-x_3)(x_0-x_2) \\ 
& + \frac{c_0+c_2}{2} (x_1+x_3)^2 + \frac{c_0-c_2}{2} (x_1-x_3)^2 \\ 

\end{aligned}
```
In the spirit of the [Butterfly Algorithm](https://en.wikipedia.org/wiki/Butterfly_diagram) for fourier transforms for 1 step would reduce number of multiples down to $\frac{N^2}{2}+N$.


</details>


**Scale.** Parameterize function 
```math
y=f\left( \begin{bmatrix}x_{0} & x_{1} &x_{2}\end{bmatrix} \right) =f\left(\alpha \begin{bmatrix}x_{1} & x_{2} & x_{0}\end{bmatrix} \right)
```
For any $\alpha\ne0$.

<details>

<summary> 
Solution
</summary>

With Taylor series, you'll run into a conclusion that no terms could exist and $y=a$. That is because scale invariant functions are often not smooth at $x_i=0$ so Taylor series could not capture them. Let us instead look into Laurent Series

```math
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{0} & x_{1} &x_{2}\end{bmatrix} \right) \\
=&\sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} \sum_{k=-\infty}^{\infty} c_{ijk} x_0^i x_1^j x_2^k 
\end{aligned}
```

Applying the invariant constraint

```math
\begin{aligned}
\sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} \sum_{k=-\infty}^{\infty} c_{ijk} x_0^i x_1^j x_2^k 
=
\sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} \sum_{k=-\infty}^{\infty} \alpha^{i+j+k} c_{ijk} x_0^i x_1^j x_2^k 
\end{aligned}
```

This only holds when the coefficients match, that is for any $(i,j,k)$
```math
\begin{aligned}
c_{ijk}=\alpha^{i+j+k} c_{ijk}
\end{aligned}
```
That is for any $(i,j,k)$, either $c_{ijk}=0$ or $i+j+k=0$.




</details>

**1D permutation with latent.** Parameterize function 
```math
y=f\left( \begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right) =f\left( \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}\begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right)
```

<details>

<summary> 
Solution
</summary>

```math
\begin{aligned}
g\left(\begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix}\right)
= &
a
+b \sum_{i=0}^{1} \sum_{j=0}^{1} x_{ij}
+c \sum_{i=0}^{1} \sum_{j=0}^{1} x_{ij}  x_{ij}
+d \sum_{i=0}^{1} \sum_{j=0}^{1} \sum_{k=0}^{1} x_{ij}  x_{ik}
+e \sum_{i=0}^{1} \sum_{j=0}^{1} \sum_{k=0}^{1} x_{ij}  x_{kj} \\
&+f \sum_{i=0}^{1} \sum_{j=0}^{1} \sum_{k=0}^{1} \sum_{l=0}^{1} x_{ij}  x_{kl}
+\ldots
\end{aligned}
```

</details>

**2D permutation.** Parameterize function 
```math
y=f\left( \begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right) =f\left( \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}\begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right) = f\left(\begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix}\begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix} \right)
```

<details>

<summary> 
Solution
</summary>

```math
\begin{aligned}
g\left(\begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix}\right)
= &
a
+b \sum_{i=0}^{1} \sum_{j=0}^{1} x_{ij}
+c \sum_{i=0}^{1} \sum_{j=0}^{1} x_{ij}  x_{ij}
+d \sum_{i=0}^{1} \sum_{j=0}^{1} \sum_{k=0}^{1} x_{ij}  x_{ik}
+e \sum_{i=0}^{1} \sum_{j=0}^{1} \sum_{k=0}^{1} x_{ij}  x_{kj} \\
&+f \sum_{i=0}^{1} \sum_{j=0}^{1} \sum_{k=0}^{1} \sum_{l=0}^{1} x_{ij}  x_{kl}
+\ldots
\end{aligned}
```

</details>


**1D permutation equivariance.** Parameterize function 
```math
\begin{bmatrix}y_{0} \\ y_{1} \\y_{2}\end{bmatrix}=F\left( \begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} \right) =F\left( \begin{bmatrix}&&\\&P&\\&&\end{bmatrix}\begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} \right)
```
For any permutation $P$.


<details>

<summary> 
Solution
</summary>

```math
\begin{aligned}
g\left(\begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix}\right)
= &
a
+b \sum_{i=0}^{1} \sum_{j=0}^{1} x_{ij}
+c \sum_{i=0}^{1} \sum_{j=0}^{1} x_{ij}  x_{ij}
+d \sum_{i=0}^{1} \sum_{j=0}^{1} \sum_{k=0}^{1} x_{ij}  x_{ik}
+e \sum_{i=0}^{1} \sum_{j=0}^{1} \sum_{k=0}^{1} x_{ij}  x_{kj} \\
&+f \sum_{i=0}^{1} \sum_{j=0}^{1} \sum_{k=0}^{1} \sum_{l=0}^{1} x_{ij}  x_{kl}
+\ldots
\end{aligned}
```

</details>


### I.3 What we have learned so far

In this section, we have learned that
1) Symmetry constraints reduce number of free parameters.
2) A Taylor-series technique can be used to parameterize symmetric functions.
3) Certain parameterizations can reduce compute.
4) Different symmetries have different impact on degrees of freedom.




## II n-D permutation symmetry
### II.1 Common types of permutation symmetry
### II.2 Permutation symmetry and einsum
### II.3 Putting everything together: Equivariant Einsum Net






## III Use cases

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

Another important. Our finding echoes an observation made by a recent work, that symmetric networks, while having fewer parameters, may not reduce compute. 


Designing general equivariant / invariant networks has been a hot topic. In fact, prior works [Max welling] discussed how to parameterize a linear layer given arbitrary linear transformations. ....  
Many network designs for specific types of nD permutation symmetry have also been proposed previously, such as . The use cases in our post are inspired by the [] work and a comment in its reviews. 
We'd like to encourage curious readers to read those works as well.






Although simple as it seems, 

## Related readings

### Other neural net designs for nD permutation symmetry

### Designing neural nets with symmetry

### Designing neural nets using einsum pooling


