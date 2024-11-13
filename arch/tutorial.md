# Einsum Pooling: nD permutation symmetric neural nets from the ground up

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
But here we'll introduce a simple yet general Taylor series-based technique necessary for studying complex equivariance patterns. Specifically, we need to enable efficient universal learners -- that can represent any such invariant or equivariant functions. 
For advanced readers, the general idea follows [Equivariant Multilayer Perceptrons (EMLP)](https://github.com/mfinzi/equivariant-MLP) with a high-order twist. 

Given the desired input-output shapes and symmetry constraints, we would proceed as the following: 
1) Express a general function that matches the input-output shapes in Taylor series form, 
2) Map the symmetry constraints into equations about the Taylor series coefficients,
3) Solve the equations for free parameters and the parameter sharing patterns, and parameterize the function using the free parameters,
4) Simplify the parameterization for efficient computation.

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
For our Taylor series form, because of the uniqueness of Taylor series, all order-k coefficients on the left hand side need to match the corresponding order-k coefficients on the right hand side. That is for any permutation matrix $P$ we have

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
These equations are all linear equations about coefficients $a$, $b_i$ and $c_{ij}$. So we can just enumerate all $P$ to get all the equations, and then solve them. For $b_i$ for example, enumerating different permutations $P$ would give

```math
\begin{bmatrix} b_0 \\ b_1 \\ b_2\end{bmatrix}  = 
\begin{bmatrix} b_0 \\ b_2 \\ b_1\end{bmatrix}  = 
\begin{bmatrix} b_1 \\ b_0 \\ b_2\end{bmatrix}  = 
\begin{bmatrix} b_1 \\ b_2 \\ b_0\end{bmatrix}  = 
\begin{bmatrix} b_2 \\ b_0 \\ b_1\end{bmatrix}  = 
\begin{bmatrix} b_2 \\ b_1 \\ b_0\end{bmatrix} 
```

That is more than enough to say $b_0=b_1=b_2$. So the order-1 term has only 1 degree of freedom.

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
For a total of 4 free parameters up to order 2, instead of 13 free parameters without the invariance constraint. More generally, for $N$ inputs, we still only need 4 parameters to express any permutation invariant function, whereas a non-invariant function needs $N^2+N+1$ parameters. In practice, parameterizing with symmetry often **reduces parameter count** exponentially. 

We can further simplify by focusing on the free parameters
```math
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{0} & x_{1} & x_{2}\end{bmatrix} \right)\\
= & a + b \sum_i x_i + (c_0-c_1) \sum_i x_i^2 + c_1 \sum_{i} \sum_{j} x_i x_j + \ldots \\
= & a + b \sum_i x_i + (c_0-c_1) \sum_i x_i^2 + c_1 (\sum_{i}x_i )^2 + \ldots
\end{aligned}
```
An important effect of this simplification is **reduced compute**. It now requires $O(N)$ compute for $N$ inputs instead of $O(N^2)$ for order-2.  

In math terms, the number of free parameters is the dimensionality of the null space of the symmetry equations. The free parameters can be numerically solved from the basis of this null space which is one of the many innovations in [1]. But note that as the basis is often not unique, numerical solutions can vary by a linear combination and therefore may not be compute-optimal, so further simplification is still needed.

Although we didn't unroll order-3 and higher terms because they are difficult to visualize, they can still be analyzed with the same approach. Just imagine a cube or a hypercube of parameters, apply the symmetry transformations simultaneously along all dimensions and solve for the parameter sharing pattern.

### I.2 Exercises
If you are interested in going a little deeper, test yourself on the following list of exercises and gain new insights.

**A. 1D translation.** Parameterize function 
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


**B. Scale.** Parameterize function 
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
That is only terms with $i+j+k=0$ would have non-zero coefficients. For example, $\frac{xy}{z^2}$. Within terms up to order-2, that is $i,j,k\in \left\{ -2,-1,0,1,2 \right\}$, the degrees of freedom is $19$ out of $5^3=125$ as the follows.

| (i,j,k) | DoF |
|:-------:|:---:|
| 0,0,0   | 1   |
| -1,0,1  | 6   |
| -2,0,2  | 6   |
| -2,1,1  | 3   |
| 2,-1,-1 | 3   |

The full parameterization is
```math
\begin{aligned}
f&\left( \begin{bmatrix}x_{0} & x_{1} &x_{2}\end{bmatrix} \right) \\
=&a + \sum_{i}\sum_{j\ne i} b_{ij} \frac{x_i}{x_j} + \sum_{i}\sum_{j\ne i} c_{ij} \frac{x_i^2}{x_j^2} + d_0 \frac{x_1 x_2}{x_0^2} + d_1 \frac{x_0 x_2}{x_1^2} + d_2 \frac{x_0 x_1}{x_2^2} + e_0 \frac{x_0^2}{x_1 x_2} + e_1 \frac{x_1^2}{x_0 x_2} + e_2 \frac{x_2^2}{x_0 x_1} 
\end{aligned}
```

Nevertheless, for scale invariance it is easier to reparameterize the input with
```math
z_0=\frac{x_0}{\sqrt{x_0^2+x_1^2+x_2^2}} \quad
z_1=\frac{x_1}{\sqrt{x_0^2+x_1^2+x_2^2}} \quad
z_2=\frac{x_2}{\sqrt{x_0^2+x_1^2+x_2^2}}
```
and express

```math
y=f\left( \begin{bmatrix}x_{0} & x_{1} &x_{2}\end{bmatrix} \right) =g\left(z_1,z_2\right)
```

</details>

**C. 1D permutation with latent.** Parameterize function 
```math
y=f\left( \begin{bmatrix}x_{0} & x_{1} \\ x_{2} & x_{3}\end{bmatrix} \right) =f\left( \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}\begin{bmatrix}x_{0} & x_{1} \\ x_{2} & x_{3}\end{bmatrix} \right)
```

<details>

<summary> 
Solution
</summary>

According to the equivariant constraint, the coefficients of the Taylor series satisfy
```math
\begin{aligned}
a = & a \\
\begin{bmatrix} b_0 & b_1 & b_2 & b_3\end{bmatrix}  = &
\begin{bmatrix} b_2 & b_3 & b_0 & b_1\end{bmatrix} 
\\
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02} & c_{03}\\
    c_{10} & c_{11} & c_{12} & c_{13}\\
    c_{20} & c_{21} & c_{22} & c_{23}\\
    c_{30} & c_{31} & c_{32} & c_{33}
\end{bmatrix} 
=& 
\begin{bmatrix} 
    c_{22} & c_{23} & c_{20} & c_{21} \\
    c_{32} & c_{33} & c_{30} & c_{31} \\
    c_{02} & c_{03} & c_{00} & c_{01} \\
    c_{12} & c_{13} & c_{10} & c_{11} \\
\end{bmatrix} 
\end{aligned}
```
Solving the equations give
```math
\begin{aligned}
\begin{bmatrix}b_0 & b_1\end{bmatrix}=&\begin{bmatrix}b_2 & b_3\end{bmatrix} \\

\begin{bmatrix} 
    c_{00} & c_{01} \\
    c_{10} & c_{11} \\
\end{bmatrix} 
=&
\begin{bmatrix} 
    c_{22} & c_{23} \\
    c_{32} & c_{33} \\
\end{bmatrix}  \\
\begin{bmatrix} 
    c_{02} & c_{03} \\
    c_{12} & c_{13} \\
\end{bmatrix} 
=&
\begin{bmatrix} 
    c_{20} & c_{21} \\
    c_{30} & c_{31} \\
\end{bmatrix} 
\end{aligned}
```
If we view each row of the input as a vectors, the coefficients can be partitioned into blocks that process those vectors, and the row-permutation invariant constraint leads to parameter sharing at the block level. We can parameterize

```math
\begin{aligned}
f&\left( \begin{bmatrix}x_{0} & x_{1}\end{bmatrix}, \begin{bmatrix}x_{2} & x_{3}\end{bmatrix} \right) \\
=&a + 
\begin{bmatrix}b_{0} & b_{1}\end{bmatrix}
\begin{bmatrix}x_{0}+x_2 \\ x_{1}+x_3\end{bmatrix}
+ 
\begin{bmatrix}x_{0} & x_{1}\end{bmatrix}
\begin{bmatrix}c_0 & c_1\\c_2 & c_3\end{bmatrix}
\begin{bmatrix}x_{0} \\ x_{1}\end{bmatrix}
+ 
\begin{bmatrix}x_{2} & x_{3}\end{bmatrix}
\begin{bmatrix}c_0 & c_1\\c_2 & c_3\end{bmatrix}
\begin{bmatrix}x_{2} \\ x_{3}\end{bmatrix}
\\
&+ 
\begin{bmatrix}x_{0} & x_{1}\end{bmatrix}
\begin{bmatrix}d_0 & d_1\\d_2 & d_3\end{bmatrix}
\begin{bmatrix}x_{2} \\ x_{3}\end{bmatrix}
+ 
\begin{bmatrix}x_{2} & x_{3}\end{bmatrix}
\begin{bmatrix}d_0 & d_1\\d_2 & d_3\end{bmatrix}
\begin{bmatrix}x_{0} \\ x_{1}\end{bmatrix}\\
=&a + 
\begin{bmatrix}b_{0} & b_{1}\end{bmatrix}
\begin{bmatrix}x_0+x_2 \\ x_1+x_3\end{bmatrix}
+ 
\begin{bmatrix}x_{0} & x_{1}\end{bmatrix}
\begin{bmatrix}c_0' & c_1'\\c_2' & c_3'\end{bmatrix}
\begin{bmatrix}x_{0} \\ x_{1}\end{bmatrix}
+ 
\begin{bmatrix}x_{2} & x_{3}\end{bmatrix}
\begin{bmatrix}c_0' & c_1'\\c_2' & c_3'\end{bmatrix}
\begin{bmatrix}x_{2} \\ x_{3}\end{bmatrix}\\
&+ 
\begin{bmatrix}x_0+x_2 & x_1+x_3\end{bmatrix}
\begin{bmatrix}d_0 & d_1\\d_2 & d_3\end{bmatrix}
\begin{bmatrix}x_0+x_2 \\ x_1+x_3\end{bmatrix}
\end{aligned}
```
The size of order-$k$ coefficient blocks for processing length-$H$ latent vectors is $H^k$. This is already much better than the full coefficients $(NH)^k$ for a set of $N$ vectors but stll large. Now the bread and butter of deep learning comes in, namely stacking more layers, low-rank factorization and non-linearities which we'll discuss more in Section II.

</details>

**D. 2D permutation.** Parameterize function 
```math
y=f\left( \begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right) =f\left( \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}\begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right) = f\left(\begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix}\begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix} \right)
```

<details>

<summary> 
Solution
</summary>

According to the equivariant constraint, the coefficients of the Taylor series satisfy
```math
\begin{aligned}
\begin{bmatrix} b_0 & b_1 & b_2 & b_3\end{bmatrix} = &
\begin{bmatrix} b_2 & b_3 & b_0 & b_1\end{bmatrix} =
\begin{bmatrix} b_1 & b_0 & b_3 & b_2\end{bmatrix} 
\\
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02} & c_{03}\\
    c_{10} & c_{11} & c_{12} & c_{13}\\
    c_{20} & c_{21} & c_{22} & c_{23}\\
    c_{30} & c_{31} & c_{32} & c_{33}
\end{bmatrix} 
=& 
\begin{bmatrix} 
    c_{22} & c_{23} & c_{20} & c_{21} \\
    c_{32} & c_{33} & c_{30} & c_{31} \\
    c_{02} & c_{03} & c_{00} & c_{01} \\
    c_{12} & c_{13} & c_{10} & c_{11} \\
\end{bmatrix} 
=
\begin{bmatrix} 
    c_{11} & c_{10} & c_{13} & c_{12} \\
    c_{01} & c_{00} & c_{03} & c_{02} \\
    c_{31} & c_{30} & c_{33} & c_{32} \\
    c_{21} & c_{20} & c_{23} & c_{22} \\
\end{bmatrix} 
\end{aligned}
```

Solving the equations gives the following parameterization with 6 degrees of freedom
```math
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right)\\
= & a + 
\begin{bmatrix} b & b & b &b\end{bmatrix} 
\begin{bmatrix} x_{00} \\ x_{01} \\ x_{10} \\ x_{11} \end{bmatrix} +
\begin{bmatrix} x_{00} & x_{01} & x_{10} & x_{11}\end{bmatrix}
\begin{bmatrix} 
    c_{0} & c_{1} & c_{2} & c_{3}\\
    c_{1} & c_{0} & c_{3} & c_{2}\\
    c_{2} & c_{3} & c_{0} & c_{1}\\
    c_{3} & c_{2} & c_{1} & c_{0}
\end{bmatrix} 
\begin{bmatrix} x_{00} \\ x_{01} \\ x_{10} \\ x_{11} \end{bmatrix}
+ \ldots
\end{aligned}
```

Let us perform a bit of merging and simplification
```math
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right)\\
= & a + b \sum_i \sum_j x_{ij} + (c_0-c_1-c_2-c_3) \sum_i \sum_j x_{ij}^2\\
&+ (c_1-c_3) \sum_i (\sum_j x_{ij})^2+(c_2-c_3) \sum_j (\sum_i x_{ij})^2  + c_3 (\sum_i \sum_j x_{ij})^2
+ \ldots
\end{aligned}
```

An interesting pattern emerges, that all terms involved are tensor contractions. In fact, this seems to be true for all flavors of permutation symmetry and the motivation behind Section II. Don't believe it? Try another case below!

</details>


**E. 2D joint permutation.** Parameterize function 
```math
y=f\left( \begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right) =f\left( \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}\begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}\right)
```

<details>

<summary> 
Solution
</summary>

According to the equivariant constraint, the coefficients of the Taylor series satisfy
```math
\begin{aligned}
\begin{bmatrix} b_0 & b_1 & b_2 & b_3\end{bmatrix} = &
\begin{bmatrix} b_3 & b_2 & b_1 & b_0\end{bmatrix} =
\\
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02} & c_{03}\\
    c_{10} & c_{11} & c_{12} & c_{13}\\
    c_{20} & c_{21} & c_{22} & c_{23}\\
    c_{30} & c_{31} & c_{32} & c_{33}
\end{bmatrix} 
=& 
\begin{bmatrix} 
    c_{33} & c_{32} & c_{31} & c_{30} \\
    c_{23} & c_{22} & c_{21} & c_{20} \\
    c_{13} & c_{12} & c_{11} & c_{10} \\
    c_{03} & c_{02} & c_{01} & c_{00} \\
\end{bmatrix} 
\end{aligned}
```

Solving the equations gives the following parameterization with 11 degrees of freedom
```math
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right)\\
= & a + 
\begin{bmatrix} b_0 & b_1 & b_1 &b_0\end{bmatrix} 
\begin{bmatrix} x_{00} \\ x_{01} \\ x_{10} \\ x_{11} \end{bmatrix} +
\begin{bmatrix} x_{00} & x_{01} & x_{10} & x_{11}\end{bmatrix}
\begin{bmatrix} 
    c_{0} & c_{1} & c_{2} & c_{3}\\
    c_{4} & c_{5} & c_{6} & c_{7}\\
    c_{7} & c_{6} & c_{5} & c_{4}\\
    c_{3} & c_{2} & c_{1} & c_{0}
\end{bmatrix} 
\begin{bmatrix} x_{00} \\ x_{01} \\ x_{10} \\ x_{11} \end{bmatrix}
+ \ldots
\end{aligned}
```
With Hessian symmetry, we may further have $c_1=c_4$ and $c_2=c_7$ which pushes free parameters count down to 9, still 3 more than regular 2D permutation invariance. If you squint really hard (and maybe try Exercise D), there exists a tensor contraction form:
```math
\begin{aligned}
f&\left( \{x_{ij}\} \right)\\
= & a + b_0' \sum_i x_{ii} + b_1' \sum_i \sum_j x_{ij} + 
c_0' \sum_i x_{ii}^2 
+ c_1' \sum_i x_{ii}\sum_j x_{ij} \\
&+ c_2' \sum_i x_{ii}\sum_j x_{ji} 
+ c_3' \sum_i x_{ii}\sum_j x_{jj}
+ c_5' \sum_i \sum_j x_{ij}^2
+ c_6' \sum_i \sum_j x_{ij} x_{ji}  
+ \ldots
\end{aligned}
```
What's different from regular 2D permutation invariance are terms involving diagonal and transpose. Also all tensor contractions here are at or below $O(N)$ compute for input size $\sqrt{N}\times \sqrt{N}$, which is better than $O(N^2)$ for the default Taylor series.

</details>


**F. 1D permutation equivariance.** Parameterize function 
```math
\begin{bmatrix}y_{0} \\ y_{1} \\y_{2}\end{bmatrix}=F\left( \begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} \right) =F\left( \begin{bmatrix}&&\\&P&\\&&\end{bmatrix}\begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} \right)
```
For any permutation $P$.


<details>

<summary> 
Solution
</summary>

The Taylor series up to order 1 can be expressed as
```math
\begin{aligned}
&\begin{bmatrix}y_{0} \\ y_{1} \\y_{2}\end{bmatrix}=F\left( \begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} \right) \\
&=\begin{bmatrix}a_0 \\ a_1 \\a_2\end{bmatrix}
+ 
\begin{bmatrix}
b_{00} & b_{01} & b_{02} \\ 
b_{10} & b_{11} & b_{12} \\
b_{20} & b_{21} & b_{22}
\end{bmatrix}
\begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} 
+
\ldots
\end{aligned}
```
The equivariant constraints are for any $P$
```math
\begin{bmatrix}
b_{00} & b_{01} & b_{02} \\ 
b_{10} & b_{11} & b_{12} \\
b_{20} & b_{21} & b_{22}
\end{bmatrix}
=
\begin{bmatrix}
 & & \\ 
 & P^T & \\ 
 & & \\ 
\end{bmatrix}
\begin{bmatrix}
b_{00} & b_{01} & b_{02} \\ 
b_{10} & b_{11} & b_{12} \\
b_{20} & b_{21} & b_{22}
\end{bmatrix}
\begin{bmatrix}
 & & \\ 
 & P & \\ 
 & & \\ 
\end{bmatrix}
```
Which is identical to the invariant constraints on order-2 terms. In general, the parameterization of equivariant functions up to order-k is very much the same as invariant functions up to order-(k+1). In the case of 1D permutation equivariance, the order-1 parameterization would be

```math
\begin{aligned}
&\begin{bmatrix}y_{0} \\ y_{1} \\y_{2}\end{bmatrix}=F\left( \begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} \right) \\
&=\begin{bmatrix}a \\ a \\a\end{bmatrix}
+ 
\begin{bmatrix}
b_{0} & b_{1} & b_{1} \\ 
b_{1} & b_{0} & b_{1} \\
b_{1} & b_{1} & b_{0}
\end{bmatrix}
\begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} 
+
\ldots
\end{aligned}
```
Rewriting in tensor contraction form using [einsum](https://pytorch.org/docs/stable/generated/torch.einsum.html) notations

```math
\begin{aligned}
Y=&F\left( X \right) \\
=& a \cdot \text{einsum(`i->i',X)} 
+ (b_0-b_1) \cdot \text{einsum(`i,i->i',X,X)} \\
&+ b_1 \cdot \text{einsum(`i,j->i',X,X)} 
\end{aligned} 
```


</details>


### I.3 What we have learned so far

In this section, we have learned that
1) Symmetry constraints reduce number of free parameters.
2) A Taylor-series technique can be used to parameterize symmetric functions.
3) Certain parameterizations can reduce compute.
4) Different symmetries can have different impacts on degrees of freedom.
5) Parameterization of equivariant functions are tied to parameterization of invariant functions
6) Permutation invariant and equivariant functions can be parameterized solely using tensor contraction terms.

A Taylor series parameterization is sound in theory. In practice however, functions compound and high order interactions are common. Taylor series often provides too little relevant capacity and too much irrelevant capacity to be useful. Engineering is key in the journey to create universal learners of equivariant functions. In the next chapter, we'll focus on permutation symmetry and design a family of practical invariant and equivariant networks for various flavors of permutation symmetry.

