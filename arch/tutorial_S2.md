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

Multiple dimensions, co-permutation and dependency are common themes here. To aid discussions, we also use a custom notation to describe the specific type of permutation symmetry, to capture both the input shape and the unique permutation axes. A fully independent batch dimension Z and a non-symmetric latent dimension H can be added optionally.


### II.2 Creating a permutation equivariant layer with einsum pooling

Across all types of permutation symmetry, as we learned in Section I through Taylor series parameterization of permutation equivariant functions, it turns out that tensor contractions are are all you need for parameterizing permutation invariant and equivariant layers, which can then be stacked into a deep network.

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
```math
\begin{aligned}
Y_{ij}=\text{einsum(`Zik,Zlk,Zlj->Zij')} \\
y=\text{einsum(`Zik,Zlk,Zlj->Z')}
\end{aligned}
```
Here a batch dimension Z is added to make sure the right hand side is not empty.  

</details>  

Let us use 



### II.3 Putting everything together: Equivariant Einsum Net





