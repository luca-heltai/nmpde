# Lecture 4 — Finite elements, dual bases, and conformity

````{admonition} Notation — primal and dual indices
:class: note

For a finite-dimensional space $P$, primal basis functions carry lower
indices, $\varphi_i\in P$, and dual basis functionals carry upper indices,
$\ell^i\in P'$. Their canonical pairing is $\ell^i(\varphi_j)=\delta^{i}{}_{j}$.
Coordinates have the opposite index position to their basis:

$$
p=\sum_{i=1}^N U^i\varphi_i,\qquad U^i=\ell^i(p),\qquad
f=\sum_{i=1}^N f_i\ell^i,\qquad f_i=f(\varphi_i).
$$

Thus primal coordinates $U^i$ are contravariant and dual coordinates $f_i$
are covariant; $f(p)=\sum_i f_iU^i$. For a bilinear form,
$A_{ij}=a(\varphi_j,\varphi_i)$ has two lower indices, and the discrete
equations read $\sum_j A_{ij}U^j=f_i$. No inner product is needed for these
identities, and indices are not raised or lowered implicitly.

Here we write sums explicitly, but Einstein summation convention is often implied. Cell labels, geometric node and edge labels,
and the mesh parameter $h$ are not tensor indices: in $\varphi_{K,i}$ and
$\ell_K^i$, $K$ labels the cell while $i$ labels the primal or dual basis.
Likewise, $U^{g_K(i)}$ is the global primal coordinate selected by a
local-to-global index map. Superscripts denoting polynomial degree or
derivative order retain their usual meaning.
````

## Objectives and prerequisites

By the end of this lecture, you should be able to define a finite element,
verify unisolvence, construct its nodal basis, and explain how local degrees
of freedom determine a global conforming space. The core examples are
$\mathbb P^k$ elements on triangles and tensor-product $\mathbb Q^k$ elements
on quadrilaterals. An optional extension develops Raviart–Thomas elements,
normal-flux degrees of freedom and the contravariant Piola transformation.

We use finite-dimensional linear algebra, Sobolev spaces and affine maps from
[Lecture 2](lecture-02.md). In [Lecture 3](lecture-03.md), the approximation
argument assumed an interpolant with polynomial reproduction and compatible
face traces. We now construct such an operator for standard Lagrange elements.

The main scalar examples are triangles and squares in two dimensions. General
simplex and tensor-product formulas are included below. Polynomial degrees are fixed throughout; no
estimate is asserted to be uniform in the degree. Meshes are finite and
conforming, with matching complete faces and no hanging nodes.

## 1. Coordinates and the dual space

Let $P$ be a real vector space of functions with dimension $N$. Given a basis
$\{\varphi_j\}_{j=1}^N$, every $p\in P$ has a unique expansion

$$
p=\sum_{j=1}^N U^j\varphi_j.
$$

The coordinate maps $\ell^i:P\to\mathbb R$, defined by $\ell^i(p)=U^i$,
are linear functionals. They form the dual basis of $P'$:

(eq:lecture-04-duality)=
$$
\ell^i(\varphi_j)=\delta^{i}{}_{j}.
$$

Here $P'$ is the algebraic dual; on a finite-dimensional normed space every
linear functional is continuous. Conversely, a basis of $P'$ determines a
unique dual basis of $P$ through this identity.

Keep the three objects distinct:

| Object | Mathematical type | Role |
|---|---|---|
| $\ell^i$ | Linear functional in $P'$ | Measures one coordinate |
| $\varphi_i$ | Function in $P$ | Reconstructs the function from coordinates |
| $U^i=\ell^i(p)$ | Real number | Value of the degree of freedom |

In finite element terminology, *degree of freedom* can refer to either the
functional $\ell^i$ or its value $U^i$; the argument must make the meaning clear.
We call $\ell^i$ a **nodal functional** and $\varphi_i$ a **nodal basis
function** or **shape function**. The adjective “nodal” in this algebraic
identity does not require a point evaluation.

A dual basis is determined by a chosen primal basis. It does not require an
inner product or a Riesz identification of $P$ with $P'$. In particular,
$\ell^i(\varphi_j)=\delta^{i}{}_{j}$ does not assert that the $\varphi_i$ are
orthonormal in $L^2$.

## 2. A finite element and unisolvence

```{prf:definition} Finite element
:label: def:lecture-04-element

A finite element is a triple $(K,P,\Sigma)$, where:

1. $K\subset\mathbb R^d$ is a bounded cell, understood with its closure,
   with nonempty interior and Lipschitz boundary;
2. $P$ is an $N$-dimensional space of scalar or vector functions on $K$;
3. $\Sigma=\{\ell^1,\ldots,\ell^N\}$ is a basis of $P'$.

In the examples below, $P$ consists of polynomials. This is the classical
finite element triple; see {cite:ts}`Ciarlet1978`.
```

Usually we choose $P$ and propose $N$ functionals first. We must then verify
that they form a basis of $P'$. Define the **degree-of-freedom map** (DoF map)

$$
\Sigma:P\to\mathbb R^N,\qquad
\Sigma(p)=(\ell^1(p),\ldots,\ell^N(p)).
$$

We use $\Sigma$ for both the family of functionals and its associated DoF map;
the context distinguishes the two meanings.

```{prf:proposition} Equivalent tests for unisolvence
:label: prop:lecture-04-unisolvence

Suppose $\dim P=N$ and $\ell^1,\ldots,\ell^N$ are linear on $P$.
The following conditions are equivalent:

1. $\Sigma$ is a basis of $P'$;
2. the DoF map $\Sigma$ is bijective;
3. if $\ell^i(p)=0$ for every $i$, then $p=0$;
4. for every $U\in\mathbb R^N$, there is a unique $p\in P$ with
   $\ell^i(p)=U^i$ for every $i$.

These conditions are called **unisolvence**.
```

**Proof.** In any basis $\{\psi_j\}_{j=1}^N$ of $P$, the matrix of the DoF map is
$M^{i}{}_{j}=\ell^i(\psi_j)$. Condition 1 says its rows are linearly independent;
condition 3 says its kernel is zero. For a square matrix, each is equivalent
to invertibility, which gives conditions 2 and 4. The corresponding nodal basis is

$$
\varphi_j=\sum_{r=1}^N\psi_r(M^{-1})^{r}{}_{j},\qquad
p=\sum_{j=1}^N\ell^j(p)\varphi_j.
$$

Thus unisolvence gives both existence and uniqueness of the basis satisfying
{eq}`eq:lecture-04-duality`. Counting the functionals establishes only that
$M$ is square.

For example, three evaluations at distinct collinear points of a triangle
are not unisolvent for $\mathbb P^1$: a nonzero affine polynomial vanishing
on their line lies in the kernel. Three noncollinear points are unisolvent,
because the zero set of a nonzero affine polynomial cannot contain them all.

### The domain of an interpolation operator

Suppose each $\ell^i$ is also defined on a function space $X$ containing $P$.
Then the canonical interpolant is

(eq:lecture-04-interpolant)=
$$
I_K:X\to P,\qquad I_Kv=\sum_{i=1}^N\ell^i(v)\varphi_i.
$$

It satisfies $\ell^j(I_Kv)=\ell^j(v)$ and $I_Kp=p$ for $p\in P$.
Consequently $I_K^2=I_K$. If $X$ is normed and the extended functionals are
bounded, then, for any fixed norm on $P$,

$$
\|I_Kv\|_P\leq
\left(\sum_i\|\ell^i\|_{X'}\|\varphi_i\|_P\right)\|v\|_X.
$$

This bound concerns a fixed element. Uniform bounds on a mesh require the
reference-element and scaling arguments of Lecture 3.

The definition of a finite element only specifies the functionals on $P$.
Their extensions to a larger space require a separate choice and domain.
For point evaluation, $X=C^0(\overline K)$ with the supremum norm is suitable.
The Sobolev embedding used in Lecture 3 also permits $X=W^{m,p}(K)$ when
$mp>d$, for integer $m\geq1$ and $1\leq p<\infty$ on the cells considered here.
Point values are not defined on arbitrary $H^1$ equivalence classes in 2D.

## 3. Lagrange elements on intervals and triangles

A **Lagrange element** uses evaluations $\ell^i(p)=p(a_i)$ at a set of
unisolvent support points $a_i\in\overline K$.

### 3.1 An interval

On $K=[0,1]$, let $P=\mathbb P^k(K)$ and choose $k+1$ distinct nodes
$a_0,\ldots,a_k$. The Lagrange polynomials

$$
\varphi_i(x)=\prod_{\substack{0\leq j\leq k\\j\neq i}}
\frac{x-a_j}{a_i-a_j}
$$

satisfy $\ell^j(\varphi_i)=\varphi_i(a_j)=\delta^{j}{}_{i}$. They are linearly independent by
evaluation at the nodes, and their number equals $\dim\mathbb P^k=k+1$.
They therefore form a basis, proving unisolvence. Equivalently, a polynomial
of degree at most $k$ with $k+1$ distinct roots is zero.

For a continuous global construction with $k\geq1$, include both endpoints
among the nodes. The remaining $k-1$ nodes are internal to the interval.

### 3.2 Barycentric coordinates and the linear triangle

Let $K$ be a nondegenerate triangle with vertices $a_0,a_1,a_2$. Its
barycentric coordinates $\lambda_0,\lambda_1,\lambda_2\in\mathbb P^1(K)$
are determined by

$$
\lambda_i(a_j)=\delta^{j}{}_{i},\qquad
\sum_{i=0}^2\lambda_i=1,\qquad
x=\sum_{i=0}^2\lambda_i(x)a_i.
$$

Existence and uniqueness follow by solving the nonsingular affine coordinate
system associated with the three noncollinear vertices. On
$\widehat K=\operatorname{conv}\{(0,0),(1,0),(0,1)\}$,

$$
\widehat\lambda_0=1-\widehat x-\widehat y,\qquad
\widehat\lambda_1=\widehat x,\qquad
\widehat\lambda_2=\widehat y.
$$

For $P=\mathbb P^1(K)$ and vertex evaluations, the shape functions are
$\varphi_i=\lambda_i$. Their nodal values prove unisolvence directly.

```{admonition} Terminology — barycentric coordinates and basis functions
:class: note

The term **barycentric coordinates** describes the geometric role of the
functions $\lambda_i$: their values $\lambda_i(x)$ are the weights expressing
a point $x$ as an affine combination of the vertices. The same functions
form the nodal basis of $\mathbb P^1(K)$ and are therefore also called
**barycentric basis functions**. We use “barycentric coordinates” when
introducing their geometric properties and “basis functions” when discussing
the finite element expansion.

These two uses of “coordinates” concern different objects. For an affine
function $p\in\mathbb P^1(K)$, its primal coordinates are the numbers
$p^i=\ell^i(p)=p(a_i)$, and its expansion is
$p=\sum_{i=0}^2p^i\lambda_i$. The functions $\lambda_i$ belong to the primal
space; the evaluations $\ell^i$ belong to its dual. The conventional lower
index in $\lambda_i(x)$ labels the vertex and the corresponding basis
function; coefficients in a primal function expansion retain upper indices.
```

### 3.3 The quadratic triangle

For $P=\mathbb P^2(K)$, use the three vertices and the three edge midpoints
$m_{ij}=(a_i+a_j)/2$. There are six nodes and
$\dim\mathbb P^2(K)=6$. The six functions

(eq:lecture-04-p2-basis)=
$$
\varphi_i=\lambda_i(2\lambda_i-1)\quad(0\leq i\leq2),\qquad
\varphi_{ij}=4\lambda_i\lambda_j\quad(0\leq i<j\leq2)
$$

have the Kronecker nodal values: at a vertex one barycentric coordinate is
one and the others vanish; at an edge midpoint the two corresponding
coordinates are $1/2$. Evaluation therefore proves linear independence,
and dimension gives a basis and unisolvence.

There is also a useful kernel proof. If $p\in\mathbb P^2(K)$ vanishes at all
six nodes, its restriction to every edge is a quadratic with three distinct
zeros, hence vanishes on that edge. Vanishing on the line $\lambda_i=0$
implies divisibility by $\lambda_i$: in affine coordinates with
$t=\lambda_i$, the coefficient of $t^0$ is zero. Applying this successively
to the three distinct edge lines would make $p$ divisible by
$\lambda_0\lambda_1\lambda_2$, a polynomial of degree three. Thus $p=0$.

The same edge argument explains conformity: a quadratic edge trace is
determined by its endpoint values and its midpoint value.

### 3.4 General simplex degrees — complete construction for independent study

Let $K$ be a nondegenerate $d$-simplex with vertices $a_0,\ldots,a_d$ and
barycentric coordinates $\lambda_0,\ldots,\lambda_d$. Fix $k\geq1$. For
multi-indices $\alpha\in\mathbb N_0^{d+1}$ with $|\alpha|=k$, set

$$
a_\alpha=\sum_{i=0}^d\frac{\alpha_i}{k}a_i,\qquad
\varphi_\alpha(x)=
\prod_{i=0}^d\left[\frac{1}{\alpha_i!}
\prod_{j=0}^{\alpha_i-1}(k\lambda_i(x)-j)\right].
$$

An empty product is one. These functions have degree at most $k$. At
$a_\beta$, if $\alpha\neq\beta$ and $|\alpha|=|\beta|=k$, some index $i$
satisfies $\beta_i<\alpha_i$. The factor with $j=\beta_i$ is then zero.
At $a_\alpha$, each bracket equals one. Thus
$\varphi_\alpha(a_\beta)=\delta^{\beta}{}_{\alpha}$.

The number of indices is $\binom{k+d}{d}=\dim\mathbb P^k(K)$, so the
functions form a basis and the nodes are unisolvent. On the face
$\lambda_i=0$, the remaining nodes have $\alpha_i=0$ and give exactly the
same construction in dimension $d-1$. They are unisolvent for
$\mathbb P^k$ on that face. A basis function with $\alpha_i>0$ vanishes there.
This proves the face property assumed in Lecture 3 for every fixed $k$.

## 4. Tensor products and reference-cell transport

### 4.1 The spaces $\mathbb P^k$ and $\mathbb Q^k$

The two common polynomial spaces are

$$
\begin{aligned}
\mathbb P^k(\mathbb R^d)&=\operatorname{span}\{x^\alpha:|\alpha|\leq k\},
&\dim\mathbb P^k&=\binom{k+d}{d},\\
\mathbb Q^k(\mathbb R^d)&=\operatorname{span}\{x^\alpha:0\leq\alpha_i\leq k\},
&\dim\mathbb Q^k&=(k+1)^d.
\end{aligned}
$$

On $\widehat K=[0,1]^d$, take the tensor grid of $k+1$ distinct 1D nodes
including the endpoints. If $L_i$ are their 1D Lagrange polynomials, then

$$
\widehat\varphi_{i_1,\ldots,i_d}(\widehat x)
=\prod_{r=1}^d L_{i_r}(\widehat x_r)
$$

has the required nodal values. The dimension count proves unisolvence.
Indeed, evaluation at the tensor node
$(a_{j_1},\ldots,a_{j_d})$ gives
$\prod_{r=1}^d\delta^{j_r}{}_{i_r}$, so the $(k+1)^d$ functions are linearly
independent and form a basis of $\mathbb Q^k$. The local interpolant is

$$
\widehat I_k v(\widehat x)
=\sum_{i_1=0}^k\cdots\sum_{i_d=0}^k
V^{i_1\ldots i_d}\widehat\varphi_{i_1,\ldots,i_d}(\widehat x),\qquad
V^{i_1\ldots i_d}=v(a_{i_1},\ldots,a_{i_d}).
$$

The corresponding dual functional is
$\widehat\ell^{i_1\ldots i_d}(v)=v(a_{i_1},\ldots,a_{i_d})$.

Restriction to a coordinate face gives the corresponding tensor basis in
dimension $d-1$. In two dimensions $Q_1$ has basis
$(1-x)(1-y),x(1-y),(1-x)y,xy$ on the unit square; these functions are bilinear.
In particular $xy\in\mathbb Q^1$ but $xy\notin\mathbb P^1$.

For $Q_2$ on the square, choose $a_0=0$, $a_1=1/2$, $a_2=1$ and write

$$
L_0(t)=2(t-1/2)(t-1),\qquad
L_1(t)=4t(1-t),\qquad
L_2(t)=2t(t-1/2).
$$

The nine shape functions are $\varphi_{ij}(x,y)=L_i(x)L_j(y)$,
$0\leq i,j\leq2$. Four nodes lie at vertices, four at edge midpoints and
one at the cell center. The center function is
$\varphi_{11}=16x(1-x)y(1-y)$ and vanishes on the entire boundary.
On the edge $y=0$, only the functions with $j=0$ survive, giving the
three 1D quadratic basis functions. Thus the three nodal values on that edge
determine its whole trace. A quadratic triangle has six local coordinates;
a biquadratic square has nine, including a cell-interior coordinate.

### 4.2 Transport of functions and functionals

Let $F_K(\widehat x)=B_K\widehat x+b_K$ be an invertible affine map from a
reference simplex or cube. Given $(\widehat K,\widehat P,\widehat\Sigma)$, define

$$
P_K=\{\widehat p\circ F_K^{-1}:\widehat p\in\widehat P\},\qquad
\ell_K^{i}(p)=\widehat\ell^i(p\circ F_K).
$$

Then $\varphi_{K,j}=\widehat\varphi_j\circ F_K^{-1}$ satisfies
$\ell_K^{i}(\varphi_{K,j})=\delta^{i}{}_{j}$. For Lagrange elements, the physical
nodes are $a_{K,i}=F_K(\widehat a_i)$. Moreover,

$$
(I_Kv)\circ F_K=\widehat I(v\circ F_K).
$$

This is the commuting identity used in the scaling proof of Lecture 3.
For simplices, affine pullback preserves $\mathbb P^k$.

For tensor elements, define the physical space by pullback of
$\mathbb Q^k(\widehat K)$. On an axis-aligned box this is the usual
$\mathbb Q^k$ space in physical coordinates. On a general mapped cell,
the pullback definition is the relevant one; a non-affine quadrilateral map
need not produce physical polynomials. Face matching also requires compatible
face parameterizations and nodal sets. We prove the global result next for
affine simplicial meshes.

## 5. Global numbering and scalar conformity

Let $\Omega$ be a bounded polygonal domain and $\mathcal T_h$ a conforming
triangulation. Use the same degree $k\geq1$ and the barycentric Lagrange
nodes on each triangle. Nodes on a shared edge coincide as sets, even when
the local vertex order is reversed.

### 5.1 Local-to-global indices

Assign one global index to each distinct physical node. Write
$g_K(i)\in\{1,\ldots,N_h\}$ for the global index of local node $i$ on $K$.
Given $U\in\mathbb R^{N_h}$, define

(eq:lecture-04-global-function)=
$$
v_h|_K=\sum_{i=1}^{\dim\mathbb P^k}U^{g_K(i)}\varphi_{K,i}.
$$

The local primal coordinates are $U_K^i=U^{g_K(i)}$, with
$\ell_K^i(v_h|_K)=U_K^i$.

For adjacent triangles $K^+,K^-$ sharing edge $e$, these two polynomials
have degree-$k$ traces and the same values at the $k+1$ edge nodes.
Unisolvence on $e$ proves equality of the traces along the entire edge.
Vertex values also agree, so the resulting function is continuous on
$\overline\Omega$.

Conversely, a continuous piecewise-$\mathbb P^k$ function has a unique value
at every global node, and local unisolvence gives the expansion above. Hence

$$
V_h=\{v\in C^0(\overline\Omega):v|_K\in\mathbb P^k(K)
\text{ for every }K\in\mathcal T_h\}.
$$

The global basis $\Phi_J$ is obtained by setting $U^J=1$ and all other
coefficients to zero. Its restriction to a cell containing node $J$ is the
corresponding local shape function, and it vanishes on other cells. Its
support is contained in the union of cells containing that node. Global
evaluations and global shape functions again satisfy the duality identity.

There are two distinct requirements in this argument: the shared face
functionals must determine the trace uniquely, and global coefficients must
identify the corresponding local measurements. A node count alone proves neither.

### 5.2 Two triangles with quadratic elements

Take a square with vertices $a=(0,0)$, $b=(1,0)$, $c=(0,1)$, $d=(1,1)$
and triangles $K_1=(a,b,c)$, $K_2=(b,d,c)$. Number the global quadratic nodes as

$$
(a,b,c,d,m_{ab},m_{bc},m_{ca},m_{bd},m_{dc})\longleftrightarrow(1,\ldots,9).
$$

Using the local order “three vertices, then three cyclic edge midpoints” gives

$$
g_{K_1}=(1,2,3,5,6,7),\qquad g_{K_2}=(2,4,3,8,9,6).
$$

The shared edge $bc$ uses indices $2,3,6$ on both cells, with local ordering
accounted for. The continuous space has dimension $9$. Keeping all twelve
local coefficients independent defines a discontinuous space of dimension $12$,
using exactly the same local polynomial spaces and support points.

On a triangular mesh, the numbers of independent nodes in the relative
interiors of mesh objects are

| Element | Per vertex | Per edge | Per triangle |
|---|---:|---:|---:|
| $P_1$ | 1 | 0 | 0 |
| $P_2$ | 1 | 1 | 0 |
| $P_3$ | 1 | 2 | 1 |
| $P_k$, $k\geq1$ | 1 | $k-1$ | $(k-1)(k-2)/2$ |

These are counts per object excluding its boundary subobjects. If $N_v,N_e,N_T$
count vertices, edges and triangles, then

$$
\dim V_h=N_v+(k-1)N_e+\frac{(k-1)(k-2)}2N_T.
$$

This connects to the “degrees of freedom per object” description used in the
next laboratory. The placement of nodes and the choice of global identification
are separate pieces of information.

### 5.3 Matching traces give $H^1$ conformity

```{prf:proposition} Scalar conforming space
:label: prop:lecture-04-h1

On the mesh above, $V_h\subset H^1(\Omega)$. Setting every boundary nodal
value to zero gives a subspace $V_{h,0}\subset H_0^1(\Omega)$.
```

**Proof.** Let $v_h\in V_h$ and $\eta\in C_c^\infty(\Omega)$. For each
coordinate $r$, integration by parts on every cell gives

$$
-\int_\Omega v_h\partial_r\eta
=\sum_K\int_K(\partial_r v_h)\eta
-\sum_K\int_{\partial K}v_h\eta\,n_{K,r}.
$$

On an interior edge, the two outward normals are opposite and the traces
of $v_h$ agree. The corresponding terms cancel. On the exterior boundary
$\eta=0$. Thus the distributional derivative is the piecewise polynomial
derivative, which belongs to $L^2(\Omega)$ on a finite mesh. This proves $H^1$
membership. If all boundary nodes vanish, each boundary edge polynomial has
$k+1$ zeros, hence is zero. The trace is zero, and the characterization of
$H_0^1$ on a Lipschitz domain yields $v_h\in H_0^1(\Omega)$.

### 5.4 Conforming tensor-product spaces

Let $\mathcal Q_h$ be a conforming mesh of axis-aligned rectangles in 2D,
without hanging nodes. Use the same fixed degree $k\geq1$ on every cell
and the equispaced reference nodes $a_i=i/k$ in each coordinate. An affine
map from $[0,1]^2$ transports the local $Q_k$ basis to each rectangle.
On a common edge, the two cells induce the same $k+1$ physical nodes.
Their edge traces are in the same 1D space $\mathbb P^k(e)$.
Identifying the corresponding nodal values therefore proves equality of
the traces, exactly as for the triangular elements.

The assembled space is

$$
V_h^Q=\{v\in C^0(\overline\Omega):
v\circ F_K\in\mathbb Q^k([0,1]^2)\text{ for every }K\in\mathcal Q_h\}.
$$

The integration-by-parts proof in §5.3 applies cell by cell to these
rectangles and shows $V_h^Q\subset H^1(\Omega)$. Zero boundary nodal values
give zero boundary edge traces, hence a subspace of $H_0^1(\Omega)$.
For more general mapped quadrilaterals, the same construction requires
compatible face maps and matching reference nodal sets under their
orientation changes.

The numbers of nodes in relative entity interiors are one per vertex,
$k-1$ per edge and $(k-1)^2$ per rectangle. Therefore, before boundary
conditions,

$$
\dim V_h^Q=N_v+(k-1)N_e+(k-1)^2N_Q,
$$

where $N_Q$ counts rectangles. In particular, two adjacent $Q_2$ squares
have $9+9-3=15$ global coordinates: three local measurements on the shared
edge are identified. Independent cell coefficients give dimension $18$.
These counts and tensor-product basis functions prepare the quadrilateral
examples in the next laboratory.

### 5.5 The global interpolant and the earlier error estimate

For $v\in C^0(\overline\Omega)$, define
$I_hv=\sum_Jv(a_J)\Phi_J$. Its restriction to $K$ is $I_Kv$.
The global dual functionals are
$\ell_h^J(v)=v(a_J)$, so $U^J=\ell_h^J(v_h)$ and
$\ell_h^I(\Phi_J)=\delta^{I}{}_{J}$.
The face argument proves $I_hv\in V_h$; zero boundary values give
$I_hv\in V_{h,0}$. This completes the construction assumed in Lecture 3.
For $u\in H^2(\Omega)\cap H_0^1(\Omega)$ in 2D, a shape-regular family
and fixed $k\geq1$ therefore give the previously proved estimate

$$
|u-u_h|_{H^1(\Omega)}
\leq C\frac M\alpha
\left(\sum_Kh_K^2|u|_{H^2(K)}^2\right)^{1/2}.
$$

Here $M$ and $\alpha$ are the continuity and coercivity constants of the
bilinear form on $H_0^1$ with the gradient norm. The interpolation constant
depends on the fixed degree and shape-regularity bound and is independent of
cell sizes. No quasi-uniformity assumption is needed for this estimate.

### 5.6 Exercises on scalar elements

1. **Global coordinates, in class.** For the two-triangle $P_2$ example,
   write the restriction of $\Phi_6$ on both triangles. Determine the
   dimension after imposing homogeneous Dirichlet data on the square boundary.
   Explain why equating only the two shared vertex values leaves a possible jump.
2. **Tensor products, in class.** On two adjacent unit squares forming
   $[0,2]\times[0,1]$, identify the shared $Q_2$ coordinates and count the
   remaining unknowns after homogeneous Dirichlet data. Write the local
   center basis function and the restrictions of the global basis function
   associated with the midpoint of the shared edge. Check their boundary traces.
3. **Unisolvence, independent study.** Verify the formula in §3.4 for a
   cubic triangle and count the nodes in each relative entity interior.

## 6. The periodic table of finite elements

The [Periodic Table of the Finite Elements, by Douglas N. Arnold and Anders
Logg](https://www-users.cse.umn.edu/~arnold/femtable/) provides a broader view.
The [full-resolution poster](https://www-users.cse.umn.edu/~arnold/femtable/femtable.pdf)
can be opened for its diagrams; the authors' site also provides the legend
and mathematical background.

Read the table by cell geometry, dimension, polynomial family and form degree.
In 3D the conformity classes correspond to $H^1$, $H(\operatorname{curl})$,
$H(\operatorname{div})$ and $L^2$. These preserve, respectively, scalar
traces, tangential traces, normal traces, or no interelement trace continuity.
The table's $k$ denotes **form degree**; our $k$ above denotes polynomial degree.
In 2D, the vector representations of the middle spaces are related by rotation.
The optional extension below develops the normal-flux representation.
A full treatment of differential forms and exact sequences is outside this lecture.

## 7. Optional extension — Raviart–Thomas elements and the Piola transformation

This section is for independent study. It extends the finite element triple
to vector fields with integral degrees of freedom, proves normal-flux
conformity and derives the contravariant Piola transformation. The core
lecture consists of §§1–6 and does not require this extension.

### 7.1 The space $H(\operatorname{div})$

For a bounded Lipschitz domain $\Omega\subset\mathbb R^2$, define

$$
H(\operatorname{div};\Omega)
=\{\boldsymbol q\in L^2(\Omega)^2:
\operatorname{div}\boldsymbol q\in L^2(\Omega)\},\qquad
\|\boldsymbol q\|_{H(\operatorname{div})}^2
=\|\boldsymbol q\|_{L^2}^2+\|\operatorname{div}\boldsymbol q\|_{L^2}^2,
$$

where divergence is distributional. This is a natural space for a flux,
such as $\boldsymbol q=-\nabla u$ in a first-order formulation of Poisson's
equation. We will not solve the mixed problem here.

For a piecewise smooth vector field, cancellation in the distributional
divergence requires matching **normal components** on interior edges. Its
tangential components can jump. This suggests degrees of freedom measuring
normal fluxes.

### 7.2 The lowest-order triangle

Let $K$ have vertices $a_0,a_1,a_2$, area $|K|>0$, and edges $e_i$ opposite
$a_i$. Let $\boldsymbol n_i$ be the outward unit normal to $e_i$. Define

(eq:lecture-04-rt-space)=
$$
\operatorname{RT}_0(K)
=\{\boldsymbol q(x)=\boldsymbol a+b x:
\boldsymbol a\in\mathbb R^2,\ b\in\mathbb R\},\qquad
\ell^i(\boldsymbol q)=\int_{e_i}\boldsymbol q\cdot\boldsymbol n_i\,\mathrm ds.
$$

The dimension is three: $(1,0)$, $(0,1)$ and $(x,y)$ form a basis. This
space is a proper subspace of $[\mathbb P^1(K)]^2$, whose dimension is six.
For $\boldsymbol q=\boldsymbol a+b x$, the normal component on a straight
edge is constant because $x\cdot\boldsymbol n_i$ is constant there.

This is the lowest-order Raviart–Thomas definition; see the
[DefElement definition](https://defelement.org/elements/raviart-thomas.html)
and its [degree-zero triangle example](https://defelement.org/elements/examples/triangle-raviart-thomas-legendre-0.html).
We use outward normals on all three edges. Basis signs on other references
can differ with their edge orientations. Our notation $\operatorname{RT}_0$
uses polynomial subdegree zero; the periodic table labels the lowest-order
2D normal-flux element $\mathrm{RT}_1$.

### 7.3 Unisolvence and an explicit dual basis

```{prf:proposition} Flux unisolvence
:label: prop:lecture-04-rt-unisolvence

The three functionals in {eq}`eq:lecture-04-rt-space` are unisolvent on
$\operatorname{RT}_0(K)$. Their dual shape functions are

$$
\boldsymbol\varphi_i(x)=\frac{x-a_i}{2|K|},\qquad i=0,1,2.
$$
```

**Kernel proof.** Suppose all three fluxes of $\boldsymbol q=\boldsymbol a+b x$
vanish. Since $\operatorname{div}\boldsymbol q=2b$, the divergence theorem gives
$2b|K|=\sum_i\ell^i(\boldsymbol q)=0$, so $b=0$. The remaining constant
vector has zero dot product with each edge normal. Two nonparallel edge
normals span $\mathbb R^2$, so $\boldsymbol a=0$.

**Basis verification.** If $j\neq i$, the vertex $a_i$ lies on the line
containing $e_j$. Thus $(x-a_i)\cdot\boldsymbol n_j=0$ on $e_j$.
On $e_i$, $(x-a_i)\cdot\boldsymbol n_i$ is the positive altitude $h_i$,
and $|e_i|h_i=2|K|$. Consequently

(eq:lecture-04-rt-duality)=
$$
\int_{e_j}\boldsymbol\varphi_i\cdot\boldsymbol n_j\,\mathrm ds
=\delta^{j}{}_{i},\qquad
\boldsymbol q=\sum_{i=0}^2\ell^i(\boldsymbol q)\boldsymbol\varphi_i.
$$

On the reference triangle with vertices $(0,0),(1,0),(0,1)$, its area is
$1/2$ and the basis is explicitly

$$
\boldsymbol\varphi_0=(x,y),\qquad
\boldsymbol\varphi_1=(x-1,y),\qquad
\boldsymbol\varphi_2=(x,y-1).
$$

For example, on the oblique edge $x+y=1$, the outward unit normal is
$(1,1)/\sqrt2$. The first function has normal component $1/\sqrt2$ and
the edge length is $\sqrt2$, giving unit flux. On the other two edges its
normal component vanishes. Each basis function has constant divergence
$1/|K|$, whose integral on $K$ is one.

The Kronecker symbol in {eq}`eq:lecture-04-rt-duality` expresses duality of
bases. The functional itself is an integral over an edge. For a general
smooth field it uses the entire normal trace, and is not a Dirac point
evaluation. On the particular polynomial space $\operatorname{RT}_0$, the
normal trace is constant, so its integral can also be computed as edge
length times its value at one edge point. This computational identity does
not replace the integral functional on the interpolation domain.

### 7.4 Oriented global fluxes give $H(\operatorname{div})$ conformity

Choose one unit normal $\boldsymbol n_e$ for each mesh edge, fixed globally.
For a cell $K$ incident to $e$, let
$s_{K,e}=\boldsymbol n_K\cdot\boldsymbol n_e\in\{-1,1\}$. The global
coordinate is $U^e=\int_e\boldsymbol q_h\cdot\boldsymbol n_e\,\mathrm ds$.
The local outward flux is $s_{K,e}U^e$, so set

$$
\boldsymbol q_h|_K=\sum_{e\subset\partial K}
s_{K,e}U^e\boldsymbol\varphi_{K,e}.
$$

The normal trace is constant on $e$. Its integral with respect to the fixed
normal is $U^e$ on both adjacent cells, so these traces agree pointwise on
the edge. In outward-normal conventions this reads

$$
\boldsymbol q_h|_{K^+}\cdot\boldsymbol n_{K^+}
+\boldsymbol q_h|_{K^-}\cdot\boldsymbol n_{K^-}=0.
$$

In particular, local outward fluxes have opposite signs across an interior
edge. Assigning them the same signed value would give the wrong matching condition.

For $\eta\in C_c^\infty(\Omega)$, cellwise integration by parts yields

$$
-\int_\Omega\boldsymbol q_h\cdot\nabla\eta
=\sum_K\int_K(\operatorname{div}\boldsymbol q_h)\eta
-\sum_K\int_{\partial K}\eta\,\boldsymbol q_h\cdot\boldsymbol n_K.
$$

The interior terms cancel by normal matching, and the exterior terms vanish
because $\eta$ has compact support. The distributional divergence is therefore
the piecewise constant divergence in $L^2$. Hence the assembled space
$Q_h\subset H(\operatorname{div};\Omega)$, with one free coordinate per edge
when no boundary flux is prescribed. Shape regularity is not needed for this
membership argument.

For a concrete tangential jump, take triangles with vertices
$(0,0),(1,0),(0,1)$ and $(-1,0),(0,0),(0,1)$, sharing the edge $x=0$.
Set $\boldsymbol q_h=(0,0)$ on the first and $(0,1)$ on the second.
Both fields are in their local $\operatorname{RT}_0$ spaces and have zero
normal component on the shared edge. The assembled field has divergence
zero and belongs to $H(\operatorname{div})$. Its second component has a
nonzero jump across that edge, so its distributional $x$-derivative contains
an edge measure; it does not belong to $H^1(\Omega)^2$.

### 7.5 Flux interpolation and a conservation identity

For $\boldsymbol v\in H^1(K)^2$, the boundary trace belongs to $L^2(\partial K)^2$,
so each edge flux is defined and bounded. Set

$$
\Pi_K^{\mathrm{RT}}\boldsymbol v
=\sum_{i=0}^2\left(\int_{e_i}\boldsymbol v\cdot\boldsymbol n_i\,\mathrm ds\right)
\boldsymbol\varphi_i.
$$

Unisolvence proves reproduction of $\operatorname{RT}_0$ and preservation of
the three fluxes. Since $\operatorname{div}\boldsymbol\varphi_i=1/|K|$,
the divergence theorem gives

(eq:lecture-04-rt-commuting)=
$$
\operatorname{div}(\Pi_K^{\mathrm{RT}}\boldsymbol v)
=\frac1{|K|}\sum_i\int_{e_i}\boldsymbol v\cdot\boldsymbol n_i\,\mathrm ds
=\frac1{|K|}\int_K\operatorname{div}\boldsymbol v\,\mathrm dx
=P_{0,K}(\operatorname{div}\boldsymbol v),
$$

where $P_{0,K}$ is the $L^2$ projection onto cellwise constants. The integrated
divergence is preserved exactly. This identity explains the connection
between normal-flux coordinates and local conservation.

Here the interpolation domain is explicitly $H^1(K)^2$. The normal trace
of a general $H(\operatorname{div};K)$ field has a weak interpretation;
defining its separate edge moments requires additional trace analysis that
we do not use in this construction.

### 7.6 The contravariant Piola transformation

Vector flux elements require a transport that preserves normal flux. For an
affine map $F_K(\widehat x)=B\widehat x+c$ with $J=\det B>0$, define

$$
\boldsymbol q(x)=\frac1J B\widehat{\boldsymbol q}(F_K^{-1}(x)).
$$

On corresponding edges, the oriented normal surface element transforms as
$\boldsymbol n\,\mathrm ds=J B^{-T}\widehat{\boldsymbol n}\,\mathrm d\widehat s$.
Taking its dot product with $J^{-1}B\widehat{\boldsymbol q}$ gives

$$
\int_e\boldsymbol q\cdot\boldsymbol n\,\mathrm ds
=\int_{\widehat e}\widehat{\boldsymbol q}\cdot\widehat{\boldsymbol n}\,
\mathrm d\widehat s.
$$

Moreover $D_x\boldsymbol q=J^{-1}B(D_{\widehat x}\widehat{\boldsymbol q})B^{-1}$,
so taking the matrix trace proves
$\operatorname{div}\boldsymbol q=J^{-1}(\widehat{\operatorname{div}}
\widehat{\boldsymbol q})\circ F_K^{-1}$.
For $\widehat{\boldsymbol q}=\widehat{\boldsymbol a}+\widehat b\widehat x$,
the image is $J^{-1}(B\widehat{\boldsymbol a}-\widehat b c+\widehat b x)$,
which belongs to $\operatorname{RT}_0(K)$. Flux preservation and bijectivity
therefore transport the dual basis. Scalar composition alone would not
preserve these flux degrees of freedom.

### 7.7 A final flux exercise and further reading

On the reference triangle, use the edge fluxes to interpolate
$\boldsymbol v=(x^2,0)$. Compute the divergence of its Raviart–Thomas
interpolant and check {eq}`eq:lecture-04-rt-commuting`. Then write the
sign relation between the two local outward fluxes on a shared edge.

For finite element triples and scalar interpolation, see
{cite:ts}`Ciarlet1978` and {cite:ts}`BrennerScott2010`.
For Raviart–Thomas spaces, normal traces and mixed formulations, see
{cite:ts}`BoffiEtAl2013`. Mixed stability and error estimates require additional
analysis and will be treated with saddle-point problems later in the course.
