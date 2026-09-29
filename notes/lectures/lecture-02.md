# Lecture 2 — Function spaces, meshes, and affine scaling

Lecture 1 introduced the variational formulation of the Poisson problem and
the Ritz–Galerkin approximation. The next question is more basic:

> What does it mean for a function to be an admissible solution, and how can we
> systematically build finite-dimensional spaces that approximate it?

This lecture develops the analytical and geometrical language needed for that
construction. We introduce Lebesgue and Sobolev spaces, explain weak
derivatives, describe triangulations and shape regularity, and derive the
scaling rules induced by affine maps from a reference element. These rules are
the mechanism behind the interpolation estimates used later in the course.

The guiding principle is that a finite element calculation separates into two
layers:

1. an **analytic layer**, which measures integrability and differentiability in
   $L^p$ and Sobolev norms;
2. a **geometric layer**, which transfers functions and polynomial bases between
   a fixed reference element and every physical cell of the mesh.

The constants in the second layer are harmless only when the mesh is
shape-regular. This is why mesh geometry is part of the approximation theory,
not merely an implementation detail.

## 1. Why function spaces enter finite element methods

Let $\Omega\subset\mathbb{R}^d$ be an open, bounded domain. A classical
formulation of a boundary-value problem asks for a function $u$ satisfying a
differential equation at every point of $\Omega$. This formulation is often
too restrictive:

- the data $f$ may be integrable without being continuous;
- the exact solution may have derivatives that are not classical functions;
- the boundary of $\Omega$ may be only piecewise smooth;
- a finite element approximation is usually only piecewise polynomial and its
  higher derivatives may jump across element interfaces.

Instead of asking for pointwise differentiability, we measure the function and
its derivatives in integral norms. The weak formulation from Lecture 1 is an
example: for Poisson's equation, it is enough to control first derivatives in
$L^2(\Omega)$, which leads naturally to $H^1(\Omega)$ and
$H_0^1(\Omega)$.

Throughout this lecture, functions that agree almost everywhere are identified.
The spaces therefore contain equivalence classes of functions, rather than a
preferred pointwise representative.

## 2. Multi-indices and weak derivatives

A **multi-index** is a vector
$\alpha=(\alpha_1,\ldots,\alpha_d)\in\mathbb{N}_0^d$. We use the notation

```{math}
:label: eq:lecture-02-multi-index
|\alpha| := \alpha_1+\cdots+\alpha_d,
\qquad
D^\alpha u :=
\frac{\partial^{|\alpha|}u}
{\partial x_1^{\alpha_1}\cdots\partial x_d^{\alpha_d}}.
```

For a smooth function, this is the usual derivative. For a function that is
only locally integrable, the same notation can be defined in the sense of
distributions.

Let $u\in L^1_{\mathrm{loc}}(\Omega)$. A function
$v\in L^1_{\mathrm{loc}}(\Omega)$ is the **weak derivative** $D^\alpha u$ if

```{math}
:label: eq:lecture-02-weak-derivative
\int_\Omega v\,\varphi\,\mathrm{d}x
=(-1)^{|\alpha|}\int_\Omega u\,D^\alpha\varphi\,\mathrm{d}x
\qquad\forall \varphi\in C_0^\infty(\Omega).
```

The test function $\varphi$ is smooth and compactly supported, so no boundary
term appears. If a classical derivative exists and is locally integrable, it is
also the weak derivative. Conversely, the weak derivative extends the notion of
derivative to functions that are not classically differentiable everywhere.

### 2.1 Two one-dimensional examples

Consider $u(x)=|x|$ on $(-1,1)$. The function is not differentiable at the
origin, but it has the weak derivative

```{math}
:label: eq:lecture-02-absolute-value
u'(x)=\operatorname{sign}(x)\quad\text{for almost every }x.
```

Since $\operatorname{sign}(x)\in L^p(-1,1)$ for every finite $p$, we have
$u\in W^{1,p}(-1,1)$. The second distributional derivative is
$2\delta_0$, which is not an $L^p$-function. Therefore $u\notin
W^{2,p}(-1,1)$.

By contrast, a jump is already too singular for $W^{1,p}$. Let
$H(x)=0$ for $x<0$ and $H(x)=1$ for $x>0$. Its distributional derivative
is $\delta_0$, so $H\notin W^{1,p}(-1,1)$ for any finite $p$. This is the
analytic reason why a globally $H^1$-conforming finite element function must
be continuous across an interior edge in two dimensions.

## 3. Lebesgue spaces

For $1\leq p<\infty$, the Lebesgue space $L^p(\Omega)$ is

```{math}
:label: eq:lecture-02-lp
L^p(\Omega):=\left\{u\text{ measurable}:
\int_\Omega |u(x)|^p\,\mathrm{d}x<\infty\right\},
```

with norm

```{math}
:label: eq:lecture-02-lp-norm
\|u\|_{L^p(\Omega)}
:=\left(\int_\Omega |u(x)|^p\,\mathrm{d}x\right)^{1/p}.
```

For $p=\infty$, we use the essential supremum:

```{math}
:label: eq:lecture-02-linfty
\|u\|_{L^\infty(\Omega)}
:=\operatorname*{ess\,sup}_{x\in\Omega}|u(x)|.
```

The adjective *essential* means that changing $u$ on a set of measure zero
does not change the norm.

If $1\leq p\leq\infty$ and $q$ is its conjugate exponent,

```{math}
:label: eq:lecture-02-conjugate-exponents
\frac1p+\frac1q=1,
```

then Hölder's inequality states

```{math}
:label: eq:lecture-02-holder
\left|\int_\Omega uv\,\mathrm{d}x\right|
\leq \|u\|_{L^p(\Omega)}\|v\|_{L^q(\Omega)}.
```

For $p=q=2$, this is the Cauchy–Schwarz inequality. Minkowski's inequality

```{math}
:label: eq:lecture-02-minkowski
\|u+v\|_{L^p(\Omega)}
\leq \|u\|_{L^p(\Omega)}+\|v\|_{L^p(\Omega)}
```

makes $L^p(\Omega)$ a Banach space. The special space $L^2(\Omega)$ is a
Hilbert space with inner product

```{math}
:label: eq:lecture-02-l2-inner-product
(u,v)_{L^2(\Omega)}
:=\int_\Omega u(x)v(x)\,\mathrm{d}x.
```

For complex-valued functions, the second factor is conjugated. We use real
functions in this course unless stated otherwise.

### 3.1 A useful integrability test

Near the origin in $\mathbb{R}^d$, consider $u(x)=|x|^{-\beta}$. In polar
coordinates,

```{math}
:label: eq:lecture-02-radial-integrability
\int_{B_1(0)}|u|^p\,\mathrm{d}x
\sim \int_0^1 r^{d-1-\beta p}\,\mathrm{d}r.
```

Consequently,

```{math}
:label: eq:lecture-02-radial-integrability-condition
|x|^{-\beta}\in L^p(B_1(0))
\quad\Longleftrightarrow\quad \beta p<d.
```

This elementary calculation is a useful reminder that integrability depends on
both the strength of a singularity and the dimension of the domain.

## 4. Sobolev spaces

For an integer $k\geq0$ and $1\leq p\leq\infty$, define

```{math}
:label: eq:lecture-02-wkp
W^{k,p}(\Omega)
:=\left\{u\in L^p(\Omega):
D^\alpha u\in L^p(\Omega)\text{ for every }|\alpha|\leq k\right\},
```

where the derivatives are weak derivatives. For $p<\infty$, the standard norm
is

```{math}
:label: eq:lecture-02-wkp-norm
\|u\|_{W^{k,p}(\Omega)}
:=\left(\sum_{|\alpha|\leq k}
\|D^\alpha u\|_{L^p(\Omega)}^p\right)^{1/p}.
```

For $p=\infty$, replace the sum by a maximum:

```{math}
:label: eq:lecture-02-wkinfty-norm
\|u\|_{W^{k,\infty}(\Omega)}
:=\max_{|\alpha|\leq k}
\|D^\alpha u\|_{L^\infty(\Omega)}.
```

The order-$k$ seminorm keeps only the derivatives of exactly order $k$:

```{math}
:label: eq:lecture-02-wkp-seminorm
|u|_{W^{k,p}(\Omega)}
:=\left(\sum_{|\alpha|=k}
\|D^\alpha u\|_{L^p(\Omega)}^p\right)^{1/p}
\qquad (p<\infty).
```

For $p=\infty$, use the maximum over $|\alpha|=k$. The seminorm may vanish
for nonzero functions: if $\Omega$ is connected, then
$|u|_{W^{k,p}(\Omega)}=0$ implies that $u$ is a polynomial of degree at
most $k-1$.

The Hilbert-space case is denoted by

```{math}
:label: eq:lecture-02-hk-definition
H^k(\Omega):=W^{k,2}(\Omega).
```

Its inner product and norm are

```{math}
:label: eq:lecture-02-hk-inner-product
(u,v)_{H^k(\Omega)}
:=\sum_{|\alpha|\leq k}(D^\alpha u,D^\alpha v)_{L^2(\Omega)},
\qquad
\|u\|_{H^k(\Omega)}:=\sqrt{(u,u)_{H^k(\Omega)}}.
```

We often write $\|u\|_{k,p,\Omega}$ and $|u|_{k,p,\Omega}$ for the norm
and seminorm, and omit $\Omega$ when it is clear from the context.

### 4.1 The spaces $W_0^{k,p}$ and $H_0^1$

The space with homogeneous boundary values is defined by closure:

```{math}
:label: eq:lecture-02-w0kp
W_0^{k,p}(\Omega)
:=\overline{C_0^\infty(\Omega)}^{\|\cdot\|_{W^{k,p}(\Omega)}}.
```

In particular, $H_0^1(\Omega)=W_0^{1,2}(\Omega)$. On a bounded Lipschitz
domain, the trace theorem identifies $H_0^1(\Omega)$ with the $H^1$-functions
whose trace vanishes on $\partial\Omega$. In the weak Poisson problem of
Lecture 1, this is the correct space for both the solution and the test
functions.

Poincaré's inequality states that there is a constant $C_\Omega>0$ such that

```{math}
:label: eq:lecture-02-poincare
\|v\|_{L^2(\Omega)}
\leq C_\Omega\|\nabla v\|_{L^2(\Omega)}
\qquad\forall v\in H_0^1(\Omega).
```

Thus, on $H_0^1(\Omega)$, the seminorm

```{math}
:label: eq:lecture-02-h1-seminorm
|v|_{H^1(\Omega)}=\|\nabla v\|_{L^2(\Omega)}
```

is a norm equivalent to the full $H^1$-norm. This is exactly what allows
the Poisson bilinear form to be coercive in Lecture 1.

### 4.2 Regularity and continuity

Sobolev regularity is not the same as pointwise smoothness. A useful rule of
thumb is that $W^{k,p}$-functions become continuous when $kp>d$, although the
precise statement involves the Sobolev embedding theorems and the regularity of
the domain. In two dimensions:

- $H^1(\Omega)=W^{1,2}(\Omega)$ is generally not contained in
  $C^0(\overline{\Omega})$;
- $W^{1,p}(\Omega)\hookrightarrow C^0(\overline{\Omega})$ for $p>2$;
- $H^2(\Omega)\hookrightarrow C^0(\overline{\Omega})$.

This distinction matters for finite elements. A piecewise polynomial function
can be smooth inside each cell and still fail to belong to $H^1(\Omega)$ if it
has a jump across an interior face.

## 5. Triangulations and mesh geometry

A mesh is a finite collection of cells that covers the computational domain.
For a simplicial mesh in two dimensions, the cells are triangles; in three
dimensions, they are tetrahedra. The word *triangulation* is used more
generally here for a partition into simple cells such as triangles,
quadrilaterals, tetrahedra, or hexahedra.

We write

```{math}
:label: eq:lecture-02-triangulation
\Omega=\mathring{\overline{\bigcup_{T\in\mathcal{T}_h}T}},
```

where $\mathcal{T}_h$ is the triangulation and $T$ denotes an individual
cell. Distinct cells may intersect only in a common lower-dimensional entity:

- the empty set;
- a common vertex;
- a common edge in two dimensions;
- a common face in three dimensions.

The partition must not contain gaps or overlaps in its interior. A mesh may
approximate a curved domain by straight-sided cells, or it may use curved
geometry through an appropriate mapping. The annular domain used in Lecture 1
is a useful example: the mesh has an inner and an outer boundary, and the
finite element calculation must know which faces belong to each boundary.

### 5.1 Local and global length scales

For a cell $T$, let $h_T$ denote its diameter:

```{math}
:label: eq:lecture-02-cell-diameter
h_T:=\sup_{x,y\in T}|x-y|.
```

For a simplex, let $\rho_T$ denote its inradius, the radius of the largest
ball contained in $T$. For a general cell, one may use an analogous inscribed
radius. The global mesh size and minimum inradius are

```{math}
:label: eq:lecture-02-global-mesh-scales
h:=\max_{T\in\mathcal{T}_h}h_T,
\qquad
\rho:=\min_{T\in\mathcal{T}_h}\rho_T.
```

A family of meshes is **shape-regular** if there is a constant
$\sigma>0$, independent of the refinement level, such that

```{math}
:label: eq:lecture-02-shape-regularity
\rho_T\geq \sigma h_T
\qquad\forall T\in\mathcal{T}_h.
```

Equivalently, no cell is allowed to become arbitrarily thin relative to its
diameter. The constant $\sigma$ controls the constants in the scaling
estimates below.

A mesh is **quasi-uniform** when all its cells have comparable size, for example

```{math}
:label: eq:lecture-02-quasi-uniform
h_T\sim h
\qquad\forall T\in\mathcal{T}_h.
```

Shape regularity is compatible with local refinement: neighbouring cells may
have different sizes, as long as their shapes do not degenerate. Quasi-uniformity
is stronger and is often convenient for global estimates, but is not necessary
for adaptive finite element methods.

### 5.2 Broken norms

If a function is smooth only on each cell, we use a broken Sobolev seminorm:

```{math}
:label: eq:lecture-02-broken-seminorm
|v|_{W^{k,p}(\mathcal{T}_h)}
:=\left(\sum_{T\in\mathcal{T}_h}
|v|_{W^{k,p}(T)}^p\right)^{1/p}.
```

The corresponding broken norm is defined by summing all derivatives of order at
most $k$ cell by cell. A broken norm does not control jumps across faces. To
obtain a conforming subspace of $H^1(\Omega)$, the local polynomial pieces
must be glued with matching traces on every interior face.

## 6. The reference element and affine maps

The central implementation idea of finite elements is to define shape functions
once on a reference cell and transport them to every physical cell.

For a $d$-simplex, use the reference simplex

```{math}
:label: eq:lecture-02-reference-simplex
\widehat{T}
:=\operatorname{conv}\{0,e_1,\ldots,e_d\},
```

where $e_1,\ldots,e_d$ are the canonical basis vectors. For a triangle,
$\widehat T$ has vertices $(0,0)$, $(1,0)$, and $(0,1)$.

Let $T$ be a physical simplex with vertices
$x_0,\ldots,x_d$. The affine map $F_T:\widehat T\to T$ is

```{math}
:label: eq:lecture-02-affine-map
F_T(\widehat x)=B_T\widehat x+b_T,
\qquad
b_T=x_0,
\qquad
B_T=[x_1-x_0\;\cdots\;x_d-x_0].
```

If the vertices are not degenerate, $B_T$ is invertible and

```{math}
:label: eq:lecture-02-affine-inverse
\widehat x=F_T^{-1}(x)=B_T^{-1}(x-b_T).
```

The absolute Jacobian determinant is constant on $T$:

```{math}
:label: eq:lecture-02-jacobian
J_T:=|\det B_T|.
```

The change-of-variables formula is therefore particularly simple:

```{math}
:label: eq:lecture-02-change-of-variables
\int_T v(x)\,\mathrm{d}x
=J_T\int_{\widehat T}(v\circ F_T)(\widehat x)\,\mathrm{d}\widehat x.
```

We use a hat for a function on the reference element:

```{math}
:label: eq:lecture-02-pullback
\widehat v(\widehat x):=v(F_T(\widehat x)).
```

For first derivatives, the chain rule gives

```{math}
:label: eq:lecture-02-gradient-transform
\widehat\nabla\widehat v(\widehat x)
=B_T^T\nabla v(F_T(\widehat x)),
\qquad
\nabla v(F_T(\widehat x))
=B_T^{-T}\widehat\nabla\widehat v(\widehat x).
```

The matrix $B_T^{-T}$, not $B_T^{-1}$ itself, transforms gradients. This
transpose is easy to miss and is important when implementing local stiffness
matrices.

## 7. Scaling of $L^p$ and Sobolev norms

The change of variables already gives the exact $L^p$ scaling. If
$\widehat v=v\circ F_T$, then, for $1\leq p<\infty$,

```{math}
:label: eq:lecture-02-lp-scaling
\|\widehat v\|_{L^p(\widehat T)}
=J_T^{-1/p}\|v\|_{L^p(T)}.
```

For $p=\infty$, the Jacobian does not appear:

```{math}
:label: eq:lecture-02-linf-scaling
\|\widehat v\|_{L^\infty(\widehat T)}
=\|v\|_{L^\infty(T)}.
```

For a first derivative, the gradient transformation and the change of variables
give

```{math}
:label: eq:lecture-02-h1-scaling-general
|\widehat v|_{W^{1,p}(\widehat T)}
\leq \|B_T\|J_T^{-1/p}|v|_{W^{1,p}(T)},
```

and, in the other direction,

```{math}
:label: eq:lecture-02-h1-scaling-reverse
|v|_{W^{1,p}(T)}
\leq \|B_T^{-1}\|J_T^{1/p}
|\widehat v|_{W^{1,p}(\widehat T)}.
```

More generally, for integer $k\geq0$, affine changes of variables yield the
estimates

```{math}
:label: eq:lecture-02-wkp-scaling-general
|\widehat v|_{W^{k,p}(\widehat T)}
\lesssim \|B_T\|^kJ_T^{-1/p}|v|_{W^{k,p}(T)},
```

and

```{math}
:label: eq:lecture-02-wkp-scaling-reverse
|v|_{W^{k,p}(T)}
\lesssim \|B_T^{-1}\|^kJ_T^{1/p}
|\widehat v|_{W^{k,p}(\widehat T)}.
```

The hidden constants depend on $k$, $p$, the dimension, and the reference
element, but not on the particular physical cell. The estimates use the fact
that the derivative of an affine map is the constant matrix $B_T$. For a
non-affine map, higher derivatives of the map also enter the chain rule.

### 7.1 Shape-regular scaling

For a shape-regular family of $d$-dimensional simplices,

```{math}
:label: eq:lecture-02-matrix-scaling
\|B_T\|\lesssim h_T,
\qquad
\|B_T^{-1}\|\lesssim \rho_T^{-1},
\qquad
J_T\sim h_T^d.
```

The last relation uses shape regularity. Without it, the determinant can be much
smaller than $h_T^d$ because a cell can be long and thin.

Combining these estimates gives the familiar rule

```{math}
:label: eq:lecture-02-sobolev-scaling
|v|_{W^{k,p}(T)}
\lesssim h_T^{d/p-k}
|v\circ F_T|_{W^{k,p}(\widehat T)},
```

and the reverse inequality

```{math}
:label: eq:lecture-02-sobolev-scaling-reverse
|v\circ F_T|_{W^{k,p}(\widehat T)}
\lesssim h_T^{k-d/p}|v|_{W^{k,p}(T)}.
```

The exponent has a clear interpretation:

- $d/p$ comes from the volume scaling of the $L^p$-integral;
- $-k$ comes from differentiating $k$ times;
- the geometry is summarized by the single length scale $h_T$ only because
  the mesh is shape-regular.

For $h_T\leq1$, the lower-order terms can be absorbed into the highest-order
term, giving analogous full-norm estimates:

```{math}
:label: eq:lecture-02-full-norm-scaling
\|v\|_{W^{k,p}(T)}
\lesssim h_T^{d/p-k}
\|v\circ F_T\|_{W^{k,p}(\widehat T)}.
```

## 8. Two sanity checks

### 8.1 An interval

Let $T=[a,a+h]$ and $\widehat T=[0,1]$. The affine map is

```{math}
:label: eq:lecture-02-interval-map
F_T(\widehat x)=a+h\widehat x,
\qquad J_T=h.
```

If $\widehat v=v\circ F_T$, then

```{math}
:label: eq:lecture-02-interval-scaling
|\widehat v|_{W^{k,p}(0,1)}
=h^{k-1/p}|v|_{W^{k,p}(a,a+h)}.
```

Equivalently,

```{math}
:label: eq:lecture-02-interval-scaling-reverse
|v|_{W^{k,p}(a,a+h)}
=h^{1/p-k}|\widehat v|_{W^{k,p}(0,1)}.
```

For $k=0$, this is the volume scaling $h^{1/p}$. Every derivative adds a
factor $h^{-1}$, as expected from the chain rule.

### 8.2 The Poisson energy on one cell

For $p=2$ and $k=1$, the Dirichlet energy on one cell is

```{math}
:label: eq:lecture-02-cell-energy-scaling
\int_T|\nabla v|^2\,\mathrm{d}x
\sim h_T^{d-2}
\int_{\widehat T}|\widehat\nabla(v\circ F_T)|^2
\,\mathrm{d}\widehat x,
```

up to constants controlled by shape regularity. Therefore:

- in $d=1$, the local stiffness contribution scales like $h_T^{-1}$;
- in $d=2$, it is scale-invariant up to the shape of the cell;
- in $d=3$, it scales like $h_T$.

The local mass integral instead scales as

```{math}
:label: eq:lecture-02-cell-mass-scaling
\int_T v^2\,\mathrm{d}x
\sim h_T^d\int_{\widehat T}(v\circ F_T)^2\,\mathrm{d}\widehat x.
```

These two rules already explain important differences between mass matrices
and stiffness matrices and are useful when checking an assembly routine.

## 9. What fails without shape regularity?

Consider a family of triangles whose diameter is approximately $h_T$, but
whose altitude is $\varepsilon h_T$ with $\varepsilon\to0$. Then

```{math}
:label: eq:lecture-02-degenerate-triangle
\rho_T\approx \varepsilon h_T,
\qquad
\|B_T^{-1}\|\approx(\varepsilon h_T)^{-1}.
```

The inverse-gradient estimate contains $\|B_T^{-1}\|$, so its constant grows
like $\varepsilon^{-1}$. A formula that depends only on $h_T$ is no longer
uniform. In practice, badly shaped cells can lead to:

- large interpolation constants;
- inaccurate numerical integration;
- poorly conditioned stiffness matrices;
- slower or unreliable iterative solver convergence.

Shape regularity is therefore the geometric hypothesis that lets us replace a
matrix estimate by a scalar estimate involving $h_T$.

## 10. From scaling to finite element approximation

Let $P^k(\widehat T)$ be a polynomial space on the reference element. The
corresponding physical-cell space is

```{math}
:label: eq:lecture-02-local-polynomial-space
P^k(T):=\{\widehat v\circ F_T^{-1}:
\widehat v\in P^k(\widehat T)\}.
```

The reference element has finitely many basis functions. Once these are known,
the affine map transports them to every cell. This has three consequences:

1. **locality:** element matrices and vectors can be computed cell by cell;
2. **reuse:** the reference basis and quadrature rule are shared by all cells;
3. **scaling:** estimates on $\widehat T$ transfer to $T$ with constants
   controlled by $h_T$ and shape regularity.

The global space is assembled by identifying local degrees of freedom on shared
entities. For a continuous Lagrange space, neighbouring polynomial pieces must
have matching values on their common face. This produces a conforming space

```{math}
:label: eq:lecture-02-conforming-space
V_h\subset H^1(\Omega)
```

and makes Céa's lemma applicable. If the pieces are allowed to jump, we obtain a
broken or discontinuous space instead; that will be the subject of the later
DG lectures.

The next analytical step is to quantify how well $P^k(T)$ approximates a
smooth function. The Bramble–Hilbert lemma does exactly this on the reference
element, while the affine scaling estimates transfer the result to every cell.
This is the bridge from the abstract estimate in Lecture 1 to the concrete
$h$-dependent error bounds of Lecture 3.

## 11. Worked checklist

For a new mesh and a new finite element implementation, the following checklist
captures the logic of this lecture:

1. Identify the domain $\Omega$ and the regularity required of its boundary.
2. Choose the Sobolev space in which the weak solution is sought.
3. Specify the cells $T\in\mathcal{T}_h$, their diameters $h_T$, and their
   inradii $\rho_T$.
4. Check whether the mesh family is shape-regular and, if needed,
   quasi-uniform.
5. Choose a reference cell $\widehat T$ and write the affine map $F_T$.
6. Compute $B_T$, $B_T^{-1}$, and $J_T=|\det B_T|$.
7. Transform functions, gradients, and integrals to $\widehat T$.
8. Use the scaling rules to track the powers of $h_T$.
9. Glue local degrees of freedom only when the desired global Sobolev conformity
   is satisfied.

## Summary

- $L^p$-spaces measure integrability; $W^{k,p}$-spaces also control weak
  derivatives up to order $k$.
- $H^k=W^{k,2}$ is a Hilbert space, and $H_0^1$ is the natural energy
  space for Poisson's equation with homogeneous Dirichlet data.
- A triangulation partitions the domain into cells, while shape regularity
  prevents cells from becoming arbitrarily thin.
- Every physical simplex is an affine image of a reference simplex.
- The Jacobian controls volume scaling, while $B_T$ and $B_T^{-1}$ control
  derivative scaling.
- The factor $h_T^{d/p-k}$ is the fundamental Sobolev scaling law on a
  shape-regular mesh.
- These estimates allow reference-element calculations to become uniform
  finite element estimates on the physical mesh.

For further reading, see the discussions of Sobolev spaces and affine scaling
in {cite:ts}`BrennerScott2010` and {cite:ts}`ErnGuermond2004`, and the finite
element framework in {cite:ts}`Ciarlet1978`.
