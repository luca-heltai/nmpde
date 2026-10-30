# Lecture 5 — From local finite elements to the global system

## Objectives and prerequisites

By the end of this lecture, you should be able to distinguish mesh indices,
local DoF indices and global DoF indices; construct a conforming Lagrange space
by identifying interface measurements; assemble a Poisson matrix and load
vector from cell contributions; and explain the mapping, quadrature and
boundary operations appearing in a finite element program.

We use finite element triples and trace unisolvence from
[Lecture 4](lecture-04.md), affine transformations from
[Lecture 2](lecture-02.md), and the Galerkin formulation from
[Lecture 1](lecture-01.md). The diagrams on DPO and numbering also connect to
[Laboratory 3](../laboratories/lab-03.md).

Throughout, meshes are conforming, without hanging nodes; scalar Lagrange
elements have a fixed degree $p\geq1$ and compatible interface nodes and maps.
The main examples use affine triangles and rectangles. All illustrative
numberings below are chosen for teaching; they are independent of the
particular ordering produced by deal.II. We use zero-based indices throughout.
As in Lecture 4, primal coefficients carry upper indices: $u_h=\sum_JU^J\Phi_J$.

## 1. Local measurements determine a polynomial

A finite element is a triple $(K,P_K,\Sigma_K)$, where $P_K$ has dimension
$n_K$ and $\Sigma_K=\{\ell_K^i\}_{i=0}^{n_K-1}$ is a basis of $P_K'$.
Unisolvence means that these measurements determine every $v\in P_K$
uniquely. For a Lagrange element,

$$
\ell_K^i(v)=v(a_{K,i}),\qquad
\ell_K^i(\varphi_{K,j})=\delta^i{}_j,\qquad
v=\sum_{i=0}^{n_K-1}\ell_K^i(v)\varphi_{K,i}.
$$

The functional $\ell_K^i$, its value and its dual shape function
$\varphi_{K,i}$ are three distinct objects. Nodal interpolation of a general
function requires well-defined point values; this construction does not make
point evaluation a bounded functional on $H^1(\Omega)$ in two dimensions.

### 1.1 DPO records the attachment of local DoFs

**Degrees of freedom per object (DPO)** counts DoFs attached to the relative
interior of each geometric object. Boundary subobjects are excluded from
that object's count. For standard scalar Lagrange nodes, the counts are:

| Object | Simplicial $P_p$ | Tensor-product $Q_p$ |
|---|---:|---:|
| Vertex | $1$ | $1$ |
| Edge interior | $p-1$ | $p-1$ |
| Triangle interior | $(p-1)(p-2)/2$ | — |
| Quadrilateral interior | — | $(p-1)^2$ |
| Tetrahedron interior | $(p-1)(p-2)(p-3)/6$ | — |
| Hexahedron interior | — | $(p-1)^3$ |

These counts apply for $p\geq1$; an entity with no interior nodes contributes
zero. In three dimensions, triangles or quadrilaterals are faces and
tetrahedra or hexahedra are volume cells. In two dimensions, triangles or
quadrilaterals are cells and edges are faces.

```{figure} ../assets/lab-03/dpo-objects.svg
:alt: Q2 nodes separated into vertices, edge interiors and the cell interior; exclusive and inclusive counts are distinguished.
:width: 100%

For $Q_2$, one DoF belongs to each vertex, one to each edge interior and one
to the cell interior: $4+4+1=9$ per square. An entire edge carries three DoFs.
```

DPO is a count together with an attachment scheme, not a local ordering.
An ordering must additionally specify which local index corresponds to each
object and each DoF on it. If an edge carries several DoFs, reversing its
orientation can permute their local positions.

### 1.2 Matching interface measurements gives continuity

First consider conforming triangles with $P_p$ elements and matching
barycentric nodes. On a common edge $e$, the difference of two traces is a
polynomial of degree at most $p$. If the $p+1$ nodal values on $e$ agree,
unisolvence on the edge makes that difference zero. For tetrahedra, use
unisolvence of the triangular face trace, including its vertex, edge and
face-interior nodes. For $Q_p$ cells, use unisolvence of tensor-product face
traces, with compatible face maps and node sets.

Consequently, identifying the corresponding interface measurements gives a
continuous piecewise finite element function. Cellwise integration by parts
then cancels the interface terms and shows that its weak gradient is the
piecewise gradient in $L^2$. Thus $V_h\subset H^1(\Omega)$.

The DPO counts specify how many measurements are attached to each object.
Continuity additionally uses their location, trace unisolvence and the
identification of the corresponding coefficients. Keeping all cell
coefficients independent gives a discontinuous space, even with the same
local polynomial spaces and nodal points.

## 2. Three index systems meet in the local-to-global map

| Index system | Labels | Depends on |
|---|---|---|
| Geometric indices | Vertices, edges, triangular/quadrilateral faces, volume cells | Mesh connectivity and its storage |
| Local DoF index $i$ | A functional and its dual shape function on one cell | Finite element ordering and cell orientation |
| Global DoF index $J$ | One independent coordinate of the assembled space | Interface identification and a chosen enumeration |

To construct the global coordinates, start with all local pairs $(K,i)$.
Identify pairs that measure the same interface node, respecting face
orientations. Take the transitive closure of these identifications. Each
equivalence class represents one global DoF; enumerate the classes to define

$$
g_K:\{0,\ldots,n_K-1\}\longrightarrow\{0,\ldots,N_h-1\}.
$$

The attachment scheme uses mesh connectivity to find shared objects. The
resulting global DoF labels have their own numbering; even for $P_1$, a DoF
index need not equal the corresponding vertex index.

```{figure} ../assets/lecture-05/numbering.svg
:alt: Two Q2 intervals with mesh vertices 0, 1, 2; repeated local indices 0, 1, 2; and independently numbered global DoFs 3, 0, 4, 1, 2 from left to right. The shared endpoint has global index 4.
:width: 100%

The same physical node can have different local indices on adjacent cells.
The two local occurrences of the shared endpoint use one global coefficient.
```

In this illustrative endpoint-first local ordering,
$g_A=(3,4,0)$ and $g_B=(4,2,1)$. Hence

$$
u_h|_A=U^3\varphi_{A,0}+U^4\varphi_{A,1}+U^0\varphi_{A,2},
\qquad
u_h|_B=U^4\varphi_{B,0}+U^2\varphi_{B,1}+U^1\varphi_{B,2}.
$$

The coefficient $U^4$ is shared. For a conforming $Q_p$ mesh of quadrilaterals,
before boundary conditions,

$$
N_h=N_v+(p-1)N_e+(p-1)^2N_Q.
$$

For hexahedra the corresponding formula is
$N_h=N_v+(p-1)N_e+(p-1)^2N_f+(p-1)^3N_H$.
For tetrahedra it is
$N_h=N_v+(p-1)N_e+\frac{(p-1)(p-2)}2N_f+
\frac{(p-1)(p-2)(p-3)}6N_T$.
All geometric counts refer to distinct mesh entities, including boundary
entities; Dirichlet constraints reduce the number of free coordinates later.

**Checkpoint.** Two adjacent $Q_2$ squares have $18$ local DoF occurrences
and $15$ global DoFs. Identify the three pairs that are merged. Explain why
only identifying the shared endpoints would leave a possible edge jump.

## 3. The variational equation becomes a matrix equation

Let $\Omega$ be a bounded, connected Lipschitz domain and $f\in L^2(\Omega)$.
For homogeneous Dirichlet data, Poisson's weak problem is

$$
u\in H_0^1(\Omega):\qquad
a(u,v)=F(v)\quad\forall v\in H_0^1(\Omega),
\qquad
a(u,v)=\int_\Omega\nabla u\cdot\nabla v,\quad
F(v)=\int_\Omega fv.
$$

Choose a conforming Lagrange space $V_h$ including boundary DoFs, with basis
$\{\Phi_J\}_{J=0}^{N_h-1}$. Its zero-boundary subspace is $V_{h,0}$.
We first assemble entries on the full space and then impose the boundary
values. The Galerkin equations are tested with the basis functions associated
with free DoFs; boundary coefficients are prescribed.

Substituting $u_h=\sum_JU^J\Phi_J$ gives

$$
\sum_J A_{IJ}U^J=b_I,\qquad
A_{IJ}=a(\Phi_J,\Phi_I),\qquad b_I=F(\Phi_I),
$$

for each free test index $I$. The row is the **test** index; the column is
the **trial** index. Poisson's bilinear form is symmetric, but keeping these
roles explicit also works for nonsymmetric problems.

### 3.1 Restricting to cells makes the calculation local

For these nodal conforming spaces, $\Phi_{g_K(i)}|_K=\varphi_{K,i}$;
global basis functions whose DoFs do not occur on $K$ vanish there. Define

$$
A^K_{ij}=\int_K\nabla\varphi_{K,j}\cdot\nabla\varphi_{K,i},
\qquad b^K_i=\int_K f\varphi_{K,i}.
$$

Since integrals over $\Omega$ are sums of cell integrals,

$$
A_{IJ}=\sum_K\sum_{\substack{i,j\\g_K(i)=I,\ g_K(j)=J}}A^K_{ij},
\qquad
b_I=\sum_K\sum_{i:g_K(i)=I}b^K_i.
$$

Therefore the assembly operation is

$$
\boxed{A_{g_K(i),g_K(j)}\mathrel{+}=A^K_{ij},\qquad
b_{g_K(i)}\mathrel{+}=b^K_i.}
$$

Different cells contribute to the same entry. Assignment would overwrite
previous contributions. Initialize global arrays once, and reset the local
arrays on every cell.

### 3.2 Restriction matrices express the same operation

Define $R_K\in\mathbb R^{n_K\times N_h}$ by
$(R_K)_{iJ}=1$ if $J=g_K(i)$ and zero otherwise. Then

$$
U_K=R_KU,\qquad
A=\sum_K R_K^T A^K R_K,\qquad b=\sum_K R_K^T b^K.
$$

Here $R_K$ gathers global coefficients into a local vector, while $R_K^T$
adds local residual contributions into a global vector. In particular,
$U^TAU=\sum_K(R_KU)^TA^K(R_KU)$.

### 3.3 Cell integrals are evaluated on the reference cell

Restricting the variational form to $K$ identifies the local contribution to
assemble. To compute that contribution, we take one further step: transform
the integral to a fixed reference cell $\widehat K$. The construction is

$$
\widehat K\xrightarrow{\ F_K\ }K,\qquad
x=F_K(\widehat x),\qquad
\varphi_{K,i}(F_K(\widehat x))=\widehat\varphi_i(\widehat x).
$$

Thus the computational route is

$$
\begin{aligned}
\text{global variational form}
&\longrightarrow \text{integrals on }K\\
&\longrightarrow \text{integrals on }\widehat K\\
&\longrightarrow \text{reference quadrature sums}\\
&\longrightarrow \text{global accumulation}.
\end{aligned}
$$

The local formulas above are written in terms of $\varphi_{K,i}$ to express
the variational problem on the physical cell. In the implementation we
tabulate the reference functions $\widehat\varphi_i$ and their reference
gradients, then use $F_K$ and its Jacobian to obtain the cell contribution.
Section 5 derives this step for both the stiffness matrix and the load.

## 4. Two triangles give an assembly calculation by hand

Take the unit square with vertices $a=(0,0)$, $b=(1,0)$, $c=(0,1)$,
$d=(1,1)$ and triangles $K_1=(a,b,c)$, $K_2=(b,d,c)$.
Use $P_1$ elements, global DoF labels $(a,b,c,d)=(0,1,2,3)$ and local
vertex orders as written:

$$
g_{K_1}=(0,1,2),\qquad g_{K_2}=(1,3,2).
$$

On $K_1$, the local basis is $(1-x-y,x,y)$; on $K_2$ it is
$(1-y,x+y-1,1-x)$. Both cells have area $1/2$, so

$$
A^{K_1}=\frac12\begin{pmatrix}2&-1&-1\\-1&1&0\\-1&0&1\end{pmatrix},
\qquad
A^{K_2}=\frac12\begin{pmatrix}1&-1&0\\-1&2&-1\\0&-1&1\end{pmatrix}.
$$

For $f=1$, each $b^{K_s}=\frac16(1,1,1)^T$, because the integral of a
linear barycentric coordinate is one third of the triangle's area. Assembly gives

$$
A=\begin{pmatrix}
1&-1/2&-1/2&0\\
-1/2&1&0&-1/2\\
-1/2&0&1&-1/2\\
0&-1/2&-1/2&1
\end{pmatrix},\qquad
b=\frac16\begin{pmatrix}1\\2\\2\\1\end{pmatrix}.
$$

```{figure} ../assets/lecture-05/assembly.svg
:alt: Two triangles sharing the diagonal b-c, their local-to-global maps and contributions to global diagonal entry A11 and load entry b1. The global matrix marks cell couplings including a structural zero on the diagonal edge.
:width: 100%

At $b$, two local measurements have become one global coordinate; two cell
integrals contribute to its global equation.
```

**Checks before imposing Dirichlet data.** The matrix is symmetric,
$A\boldsymbol1=0$, and $\boldsymbol1^Tb=1=|\Omega|$. The unconstrained
stiffness matrix is positive semidefinite: constants have zero energy.
All four vertices lie on the boundary, so this particular $P_1$ patch has
no free unknowns after full Dirichlet conditions. It illustrates assembly;
a mesh with interior nodes is needed for a nontrivial Dirichlet solve.

**Exercise in class.** Find the two contributions to $A_{11}$ and $b_1$.
Then find $A_{13}$, taking care that the global index $3$ is local index $1$
on $K_2$. Finally, explain why $A_{03}=0$ without evaluating an integral.

## 5. Mapping and quadrature produce each local entry

Let $F_K:\widehat K\to K$ be a differentiable bijective cell map with
nonsingular Jacobian $J_K(\widehat x)=DF_K(\widehat x)$ and compatible
reference face maps. For scalar Lagrange elements the same reference basis
is transported to each physical cell by composition with $F_K^{-1}$.
The point $\widehat x$ and its image $x=F_K(\widehat x)$ give the same
shape-function value:

$$
\begin{array}{ccc}
\widehat K & \xrightarrow{\ F_K\ } & K\\
\widehat\varphi_i\searrow & & \swarrow\varphi_{K,i}\\
&\mathbb R&
\end{array}
\qquad
\varphi_{K,i}\circ F_K=\widehat\varphi_i.
$$

```{figure} ../assets/lecture-05/reference-map.svg
:alt: A fixed reference triangle maps to a physical triangle through F_K. A reference quadrature point maps to its physical image. Reference shape values agree at corresponding points; gradients and integration weights require Jacobian transformations.
:width: 100%

The geometry changes from cell to cell. The reference shape functions and
reference quadrature data are reused for every cell with this element type.
```

Solving for the physical shape function and applying the chain rule gives

$$
\varphi_{K,i}=\widehat\varphi_i\circ F_K^{-1},\qquad
\nabla_x\varphi_{K,i}(F_K(\widehat x))
=J_K(\widehat x)^{-T}\widehat\nabla\widehat\varphi_i(\widehat x).
$$

For every integrable physical-cell integrand $q_K$, change of variables gives

$$
\int_K q_K(x)\,dx
=\int_{\widehat K}q_K(F_K(\widehat x))
|\det J_K(\widehat x)|\,d\widehat x.
$$

Apply this identity to the two local integrands. Substituting the reference
values and transformed reference gradients yields

$$
A^K_{ij}=\int_{\widehat K}
(J_K^{-T}\widehat\nabla\widehat\varphi_j)\cdot
(J_K^{-T}\widehat\nabla\widehat\varphi_i)
|\det J_K|\,d\widehat x.
$$

$$
b^K_i=\int_{\widehat K}
f(F_K(\widehat x))\widehat\varphi_i(\widehat x)
|\det J_K(\widehat x)|\,d\widehat x.
$$

Both formulas now contain only reference shape functions
$\widehat\varphi_i$ and their reference derivatives. Cell dependence enters
through $F_K$, $J_K$ and the physical datum evaluated at $F_K(\widehat x)$.

For reference quadrature nodes and weights $(\widehat x_q,w_q)$, define
$x_q=F_K(\widehat x_q)$, $G_{i,q}=J_K(\widehat x_q)^{-T}
\widehat\nabla\widehat\varphi_i(\widehat x_q)$ and
$W_{K,q}=|\det J_K(\widehat x_q)|w_q$. Then

$$
A^{K,Q}_{ij}=\sum_q G_{j,q}\cdot G_{i,q}\,W_{K,q},\qquad
b^{K,Q}_i=\sum_q f(x_q)\widehat\varphi_i(\widehat x_q)W_{K,q}.
$$

The superscript $Q$ distinguishes quadrature approximations from the exact
integrals. The local-to-global accumulation is the same for both.

```{figure} ../assets/lecture-05/pipeline.svg
:alt: Reference shape functions and quadrature enter a mapping step that produces physical quadrature points, physical gradients and JxW. These form local matrix and load entries, which are accumulated using the local-to-global map.
:width: 100%

The finite element supplies reference functions, the mapping supplies
geometry, and the DoFHandler supplies global indices.
```

### 5.1 Quadrature exactness has geometric assumptions

For constant-coefficient Poisson on an affine simplex, the stiffness
integrand has total polynomial degree at most $2p-2$. A simplex quadrature
rule exact to that degree integrates it exactly. If $f$ is polynomial of
degree $r$, the load integrand has degree at most $r+p$.

On an axis-aligned affine box with $Q_p$ functions, tensor-product Gauss
quadrature with $p+1$ points per direction is sufficient for the stiffness
matrix. Each coordinate degree is at most $2p$, and this rule is exact
through degree $2p+1$ in each coordinate. For a constant load, $p+1$ points
also suffice. A general load must be checked separately.

For a non-affine cell map, Jacobians and inverse Jacobians vary: polynomial
degree counting on the reference basis alone does not establish exactness.
The same warning applies to variable coefficients or nonpolynomial data.
Underintegration can introduce additional zero-energy modes; symmetry alone
does not guarantee a stable discrete system.

```{admonition} The reference basis is the computational basis
:class: important

Every cell integral in this assembly is evaluated after transformation to
$\widehat K$, using $\widehat\varphi_i(\widehat x_q)$ and
$\widehat\nabla\widehat\varphi_i(\widehat x_q)$. The functions
$\varphi_{K,i}$ express the physical-cell formulation; their values and
gradients at quadrature points are obtained from these reference tables via
$F_K$. We do not construct and integrate a new physical-coordinate basis
formula separately on every cell.
```

## 6. The formulas become an assembly loop

In deal.II, the element, mapping and quadrature are combined by `FEValues`.
The following is the assembly core, assuming initialized global arrays,
a `DoFHandler`, and a scalar `rhs_function`. It is a reading guide, not a
standalone program. The local matrix entries approximate the integrals above.

```cpp
MappingQ1<dim> mapping;
QGauss<dim> quadrature(fe.degree + 1);
FEValues<dim> fe_values(mapping, fe, quadrature,
                        update_values | update_gradients |
                        update_quadrature_points | update_JxW_values);

const unsigned int n = fe.n_dofs_per_cell();
FullMatrix<double> cell_matrix(n, n);
Vector<double> cell_rhs(n);
std::vector<types::global_dof_index> indices(n);

system_matrix = 0;
system_rhs = 0;
for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_rhs = 0;
    for (const unsigned int q : fe_values.quadrature_point_indices())
      for (const unsigned int i : fe_values.dof_indices())
        {
          for (const unsigned int j : fe_values.dof_indices())
            cell_matrix(i, j) +=
              (fe_values.shape_grad(j, q) * fe_values.shape_grad(i, q)) *
              fe_values.JxW(q);
          cell_rhs(i) += rhs_function.value(fe_values.quadrature_point(q)) *
                         fe_values.shape_value(i, q) * fe_values.JxW(q);
        }

    cell->get_dof_indices(indices);
    for (unsigned int i = 0; i < n; ++i)
      {
        for (unsigned int j = 0; j < n; ++j)
          system_matrix.add(indices[i], indices[j], cell_matrix(i, j));
        system_rhs[indices[i]] += cell_rhs(i);
      }
  }
```

`MappingQ1` describes multilinear hypercube geometry; it is affine on
parallelograms/parallelepipeds and can be non-affine on general quadrilaterals
or hexahedra. Its degree is independent of the solution degree in `FE_Q`.
Use an appropriate mapping and quadrature for other reference-cell types.

| Mathematical quantity | deal.II expression |
|---|---|
| $x_q$ | `fe_values.quadrature_point(q)` |
| $\varphi_{K,i}(x_q)$ | `fe_values.shape_value(i, q)` |
| $\nabla_x\varphi_{K,i}(x_q)$ | `fe_values.shape_grad(i, q)` |
| $\lvert\det J_K(\widehat x_q)\rvert w_q$ | `fe_values.JxW(q)` |
| $g_K(i)$ | `indices[i]` after `cell->get_dof_indices(indices)` |

In particular, `shape_value(i, q)` supplies
$\widehat\varphi_i(\widehat x_q)$, which equals
$\varphi_{K,i}(F_K(\widehat x_q))$ for these scalar mapped elements.
`shape_grad(i, q)` supplies
$J_K(\widehat x_q)^{-T}\widehat\nabla\widehat\varphi_i(\widehat x_q)$.
Thus the API presents physical-cell quantities computed from the reference
basis and the cell mapping.

`shape_grad` already returns the physical gradient and `JxW` already
includes the quadrature weight and geometric volume factor. Applying either
transformation a second time gives an incorrect integral. `reinit(cell)`
updates the cell-dependent data; the local index ranges remain those of the
finite element.

### 6.1 Sparsity follows the support of the basis

If two global basis functions do not occur together on any cell, their
volume stiffness entry is zero. Before computing integrals, allocate every
pair $(g_K(i),g_K(j))$ that occurs on a cell. This gives a safe structural
pattern for the conforming volume operator, possibly including entries
whose numerical value is zero.

```cpp
DynamicSparsityPattern dsp(dof_handler.n_dofs());
DoFTools::make_sparsity_pattern(dof_handler, dsp);
sparsity_pattern.copy_from(dsp);
system_matrix.reinit(sparsity_pattern);
```

In the two-triangle example, $(1,2)$ occurs on both cells, but $A_{12}=0$
because the corresponding gradients are orthogonal on each cell.
The pair $(0,3)$ never occurs on a cell. These are different explanations
for a zero entry. Face-coupled DG operators require additional couplings;
the cell-only pattern described here concerns the conforming Poisson operator.

### 6.2 Renumbering permutes the algebra

Let $\pi$ be any permutation of global indices and set
$\widetilde U^{\pi(J)}=U^J$, $\widetilde\Phi_{\pi(J)}=\Phi_J$.
If $P_{\pi(J),J}=1$, then

$$
\widetilde U=PU,\qquad \widetilde A=PAP^T,\qquad
\widetilde b=Pb,\qquad
\widetilde g_K=\pi\circ g_K.
$$

The represented function $u_h$ is unchanged. Matrix bandwidth and memory
access patterns can change; the variational problem and the eigenvalues of
the symmetrically permuted stiffness matrix are unchanged. Permute boundary
labels and coefficient vectors consistently as well.

## 7. Dirichlet values turn the assembled matrix into a solvable system

Split the indices into free indices $F$ and prescribed boundary indices $D$.
For Lagrange elements, prescribe a finite element boundary trace with nodal
coefficients $U_D=G$. For nonpolynomial boundary data this is a discrete
approximation of the boundary datum. The free equations are

$$
\begin{pmatrix}A_{FF}&A_{FD}\end{pmatrix}
\begin{pmatrix}U_F\\G\end{pmatrix}=b_F,
\qquad
\boxed{A_{FF}U_F=b_F-A_{FD}G.}
$$

Subtracting $A_{FD}G$ accounts for the known boundary contribution. For
homogeneous data this term vanishes. Boundary rows of the unconstrained
matrix are not additional Galerkin equations to enforce: boundary test
functions do not belong to $V_{h,0}$.

For exact integration, $A_{FF}$ is symmetric positive definite when the
Dirichlet boundary removes constants (in particular, for full Dirichlet
conditions on this connected domain). Indeed, for a nonzero free coefficient
vector $X$, the corresponding $v_h\in V_{h,0}$ satisfies

$$
X^TA_{FF}X=\int_\Omega|\nabla v_h|^2>0
$$

by Poincaré's inequality and linear independence of the basis. A quadrature
matrix needs a quadrature rule that preserves this definiteness.

The basic tutorials use `VectorTools::interpolate_boundary_values` and
`MatrixTools::apply_boundary_values`. With symmetric elimination, the latter
adjusts the right-hand side and removes boundary columns as well as boundary
rows, preserving symmetry. Replacing boundary rows alone generally destroys
symmetry and is unsuitable for a conjugate-gradient solve of that modified
full matrix. Hanging-node constraints and `AffineConstraints` will be treated
in a later laboratory.

**Checkpoint.** In the two-triangle patch, prescribe the affine harmonic
function $u=x+y$. Then $G=(0,1,1,2)^T$ and $f=0$. Check that the two cell
representations agree on the diagonal. All values are prescribed in this
patch; a solve with interior nodes will test the free equations.

## 8. A reading path for the next laboratories

1. **Geometry and integration.** Identify the mapping, quadrature nodes,
   values, gradients and `JxW`. Begin [Laboratory 4](../laboratories/lab-04.md)
   with a cellwise area integral, then use the same objects to assemble Poisson.
2. **Assembly.** Follow `setup_system()` and `assemble_system()` in
   [step-3, deal.II 9.7.1](https://github.com/dealii/dealii/blob/v9.7.1/examples/step-3/step-3.cc).
   Match each operation to a formula in this lecture.
3. **Data and dimension.** In
   [step-4, deal.II 9.7.1](https://github.com/dealii/dealii/blob/v9.7.1/examples/step-4/step-4.cc),
   identify the spatially varying right-hand side, nonhomogeneous boundary
   values and dimension-dependent types. The assembly principle is unchanged.

For independent study, repeat the two-triangle assembly after swapping global
indices $0$ and $3$. Verify the permutation identities without recomputing
cell integrals. Then use a mesh with interior nodes and an affine harmonic
solution with matching Dirichlet data: exact representability, exact
integration and an accurate linear solve should reproduce its nodal values
up to roundoff and solver tolerance. This checks assembly and boundary
handling without assuming an $L^2$ convergence theorem.

Elliptic regularity, the pacman example, fractional spaces and the
Aubin–Nitsche $L^2$ error estimate are the subject of the next theoretical
lecture. Computing an $L^2$ error in a laboratory and proving its convergence
rate are separate tasks.
