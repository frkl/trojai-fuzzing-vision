# Einsum Pooling: nD permutation symmetric neural nets from the ground up


## II Engineering Networks with nD Permutation Symmetry

From matrices to sets to symbolic processing, permutation symmetry is found in many problems and requires extra attention during modeling. When handled properly however, permutation symmetry is also a blessing. As we have learned in the previous section, if parameterized properly, permutation symmetry has the potential to exponentially reduce parameter count and compute for highly efficient learning. At the other end of the spectrum, reciting the success recipe of deep learning, we can scale the latent dimension and stack equivariant layers to create exponentially more expressive networks at the same parameter count and compute as a regular network.

![Network with Equivariant Backbone](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png)


Devil's in the details, in this section we'll walk through the design of permutation symmetric neural networks for various types of permutation symmetry.


There are many places where you'll see permutation symmetry and they often come in different forms. So in Section II.1 we'll first start from a summary of common types of permutation symmetry. 
And then Section II.2 will discuss the design of permutation equivariant layers. Permutation symmetry turns out to be closely tied to tensor contractions. That would allow us to synthesize efficient high-order permutation equivariant layers automatically in a procedural manner.
And finally in Section II.3 we discuss further optimizations that helps practical implementation. 

### II.1 Common types of permutation symmetry

In the following table we list a few common problems with different types of permutation symmetry.

| Problem       | Illustration    | Symmetry type  |
| ------------- |:-------------:| -----:|
|       | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

Multiple dimensions, co-permutation and dependency are common themes here. To aid discussions, we also use a custom notation to describe the specific type of permutation symmetry, to capture both the input shape and the unique permutation axes. A fully independent batch dimension Z and a non-symmetric latent dimension H can be added optionally.


### II.2 Permutation symmetry and einsums



### II.3 Putting everything together: Equivariant Einsum Net





