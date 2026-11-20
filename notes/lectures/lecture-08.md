# Lecture 8 — Scott–Zhang interpolation and residual error estimation

## Objectives and prerequisites

This lecture completes the Scott–Zhang construction started in
[Lecture 7, §§5–6](lecture-07.md), then introduces a posteriori estimation
through the conforming Poisson problem. By the end, you should be able to

- select an averaging edge or cell for each Lagrange degree of freedom;
- explain projection, boundary preservation and the local interpolation bounds;
- derive the cell residual and the normal-flux jump from the weak formulation;
- bound the dual norm of the residual and derive a computable error estimator.

The prerequisites are traces, affine scaling, the scaled trace inequality,
conforming finite elements and Galerkin orthogonality. Local efficiency,
bubble functions and the role of data oscillation in lower bounds are reserved
for the next theoretical lecture.

**Standing assumptions.** Let $\Omega\subset\mathbb R^2$ be a bounded,
connected polygonal Lipschitz domain. Let $\mathcal T_h$ be a conforming family
of affine triangular meshes, without hanging nodes, with uniformly bounded
$h_T/\rho_T$. Here $h_T=\operatorname{diam}T$ and $\rho_T$ is the inradius.
For an edge $F$, $h_F=|F|$. Let $\mathcal F_h^i$ be the set of interior edges,
each listed once. No global quasi-uniformity is assumed.

Write $\|v\|_{0,D}^2=\int_D|v|^2$ and
$|v|_{1,D}^2=\int_D|\nabla v|^2$; $\|\cdot\|_{1,D}$ is the full norm.
An omitted domain means $\Omega$. All constants below are independent of
mesh sizes and the functions being estimated. They may depend on $\Omega$
and the shape-regularity bound; for degree $k>1$ they may also depend on the
fixed degree. The proofs of the interpolation estimates below concern $P_1$.

## 1. Averaging simplices replace point evaluations

For a fixed integer $k\geq 1$, let

$$
S_h^k=\{v_h\in C^0(\overline\Omega):v_h|_T\in P_k(T)
       \text{ for every }T\in\mathcal T_h\}.
$$

Let $a_1,\ldots,a_N$ be the global Lagrange support points and let
$\phi_1,\ldots,\phi_N$ be the corresponding nodal basis of $S_h^k$.
Scott–Zhang interpolation assigns to each $a_i$ a moment of $v$ over a
selected simplex $\Xi_i$. Cell moments use the restriction of $v$ and edge
moments use its trace. Both are defined for $v\in H^1(\Omega)$.

### 1.1 The node location determines the averaging simplex

For each global support point $a_i$, choose one simplex $\Xi_i$ according to
the following rules and fix the choice for the operator.

| Location of $a_i$ | Choice of $\Xi_i$ |
| --- | --- |
| Strictly inside a triangle $T$ | The triangle $T$ itself. The moment is an integral over the cell. |
| In the relative interior of a mesh edge $F$ | The edge $F$ itself, whether interior or on the boundary. |
| At an interior mesh vertex | Any one incident edge. |
| At a boundary mesh vertex | Any one incident boundary edge. |

Thus every boundary support point uses a simplex contained in
$\partial\Omega$. The cell case occurs for triangular Lagrange elements
of degree $k\geq 3$. For $P_1$, all support points are vertices, so
$\Xi_i=F_i$ is an edge for every $i$.

```{figure} ../assets/lecture-08/sz-simplex-choices.svg
:alt: Five node types and their averaging edges or cells, followed by the distinction between the averaging set and the support of a hat function.
:name: fig:lecture-08-sz-choices

Teal marks the selected averaging simplex $\Xi_i$. Dashed orange edges mark
other admissible choices. For a point strictly inside a triangle, the entire
triangle is selected. In the last panel the blue region is
$\operatorname{supp}\phi_i$ and the teal segment is $\Xi_i$.
```

### 1.2 A dual moment defines each nodal coefficient

Let $I_i=\{j:a_j\in\Xi_i\}$. The restrictions
$\{\phi_j|_{\Xi_i}:j\in I_i\}$ form the Lagrange basis of $P_k(\Xi_i)$.
For $j\notin I_i$, $\phi_j|_{\Xi_i}=0$ by nodal unisolvence on $\Xi_i$.
Define $\psi_i\in P_k(\Xi_i)$ by

$$
\int_{\Xi_i}\psi_i\,\phi_j|_{\Xi_i}\,d\mu_i=\delta_{ij}
\qquad\text{for every }j\in I_i,
$$

where $d\mu_i$ is area measure on a triangle and arclength on an edge.
The corresponding mass matrix is the Gram matrix of a basis under the
$L^2(\Xi_i)$ inner product. It is positive definite, so $\psi_i$ is unique.

Let $\gamma_{\Xi_i}v$ denote $v|_T$ when $\Xi_i=T$ and the Sobolev trace of
$v$ when $\Xi_i$ is an edge. On an interior edge the traces from its two
adjacent cells agree for every $v\in H^1(\Omega)$. Define the linear operator
$J_h:H^1(\Omega)\to S_h^k$ by

```{math}
:label: eq:lecture-08-sz-definition
\ell_i(v)=\int_{\Xi_i}\psi_i\,\gamma_{\Xi_i}v\,d\mu_i,
\qquad J_hv=\sum_{i=1}^N\ell_i(v)\phi_i.
```

### 1.3 Projection and boundary preservation follow from duality

For $v_h=\sum_{j=1}^N c_j\phi_j\in S_h^k$, its restriction to $\Xi_i$ is
$\sum_{j\in I_i}c_j\phi_j|_{\Xi_i}$. Duality gives $\ell_i(v_h)=c_i$ and hence

$$
J_hv_h=v_h,\qquad J_h^2=J_h.
$$

In particular, $J_h$ reproduces every global polynomial of degree at most $k$.

If $v\in H_0^1(\Omega)$, each boundary support point has $\ell_i(v)=0$
because its moment uses the zero trace of $v$ on a boundary edge.
On each boundary edge, the polynomial trace of $J_hv$ vanishes at all its
Lagrange support points, so it vanishes identically. Thus

$$
J_h(H_0^1(\Omega))\subset S_h^k\cap H_0^1(\Omega).
$$

For representable data $\gamma v=g_h\in\gamma S_h^k$, choose $z_h\in S_h^k$
with trace $g_h$. The boundary moments of $v$ and $z_h$ agree, so $J_hv$
and $J_hz_h=z_h$ have the same trace. For general boundary data the trace
of $J_hv$ is a moment approximation determined by $\gamma v$.
The general theory is developed in {cite:ts}`ScottZhang1990`.

### 1.4 A complete cubic example

```{figure} ../assets/lecture-08/sz-p3-all-dofs.svg
:alt: Eight triangular cells with sixteen edges, all forty-nine cubic Lagrange nodes, and a complete table assigning one averaging edge or cell to every global node.
:name: fig:lecture-08-sz-all-dofs

One complete $P_3$ assignment with 9 vertex nodes, 32 edge-interior nodes
and 8 cell-interior nodes. Each of the latter uses its own triangle.
$F_j$ labels an edge and $T_j$ a triangle. The table specifies $\Xi_i$ for
every global node $a_i$.
```

## 2. Local estimates identify the relevant mesh weights

From here on, take $k=1$ and write $S_h=S_h^1$ and
$V_h=S_h\cap H_0^1(\Omega)$.

### 2.1 The patch contains all averaging edges needed on a cell

For $T\in\mathcal T_h$, let $\omega_T$ be the interior relative to $\Omega$
of the union of the closed triangles sharing at least one vertex with $T$.
Only the coefficients at the three vertices of $T$ affect $J_hv|_T$.
All their selected edges are contained in $\overline\omega_T$.

```{figure} ../assets/lecture-08/sz-locality-patch.svg
:alt: A target triangle and its vertex patch, followed by three selected averaging edges contained in the patch and possibly outside the target cell.
:name: fig:lecture-08-sz-patch

The averaging edges can lie outside $T$. The larger patch $\omega_T$ controls
all three coefficients and is the domain used in the local estimates.
```

Shape regularity bounds the number of triangles in a vertex star.
Consecutive triangles in the star share an edge and have comparable sizes;
the chain length is uniformly bounded. Thus every cell in $\omega_T$ has size
comparable to $h_T$, and the patches have a uniformly bounded overlap.
These are local properties, compatible with global mesh grading.

### 2.2 A moment bound gives stability after subtraction of constants

For each vertex of $T$, choose a triangle $K_i$ adjacent to its selected
edge $F_i$, with $K_i\subset\omega_T$, and set $L_i=|F_i|$. The edge
dual satisfies $\|\psi_i\|_{0,F_i}=2L_i^{-1/2}$ by
[Lecture 7, §5.1](lecture-07.md). Cauchy–Schwarz and the scaled trace
inequality give

$$
\begin{aligned}
|\ell_i(v)|&\leq 2L_i^{-1/2}\|v\|_{0,F_i}\\
&\leq C\bigl(h_T^{-1}\|v\|_{0,K_i}+|v|_{1,K_i}\bigr).
\end{aligned}
$$

The hat-function bounds $\|\phi_i\|_{0,T}\leq Ch_T$ and
$|\phi_i|_{1,T}\leq C$ therefore imply

```{math}
:label: eq:lecture-08-preliminary-sz
\begin{aligned}
\|J_hv\|_{0,T}&\leq
C\bigl(\|v\|_{0,\omega_T}+h_T|v|_{1,\omega_T}\bigr),\\
|J_hv|_{1,T}&\leq
C\bigl(h_T^{-1}\|v\|_{0,\omega_T}+|v|_{1,\omega_T}\bigr).
\end{aligned}
```

Choose a constant $c$ satisfying the patch Poincaré inequality
$\|v-c\|_{0,\omega_T}\leq Ch_T|v|_{1,\omega_T}$.
Its uniformity follows by comparing cell means along the bounded chains of
edge neighbors; the full argument is in [Lecture 7, §6.2](lecture-07.md).
Since $J_hc=c$, applying the second bound to $v-c$ gives

$$
|J_hv|_{1,T}=|J_h(v-c)|_{1,T}\leq C|v|_{1,\omega_T}.
$$

Summing over $T$, using bounded overlap and the first bound above, also gives
$\|J_hv\|_1\leq C\|v\|_1$. The constant can depend on the fixed domain size.

### 2.3 Cell and edge approximation bounds

```{prf:theorem} The interpolation estimates used by the residual estimator
:label: thm:lecture-08-sz-residual

For $v\in H^1(\Omega)$ and every $T\in\mathcal T_h$,

$$
\|v-J_hv\|_{0,T}\leq Ch_T|v|_{1,\omega_T}.
$$

For an interior edge $F$, choose one adjacent cell $T(F)$ and define
$\omega_F=\omega_{T(F)}$. Then

$$
\|v-J_hv\|_{0,F}\leq Ch_F^{1/2}|v|_{1,\omega_F}.
$$

Both constants are independent of mesh sizes. The family of these cell and
edge patches has uniformly bounded overlap.
```

**Proof.** With the constant $c$ from the patch Poincaré inequality and
$J_hc=c$, the first bound in {eq}`eq:lecture-08-preliminary-sz` gives

$$
\begin{aligned}
\|v-J_hv\|_{0,T}
&\leq \|v-c\|_{0,T}+\|J_h(v-c)\|_{0,T}\\
&\leq C\bigl(\|v-c\|_{0,\omega_T}+h_T|v|_{1,\omega_T}\bigr)\\
&\leq Ch_T|v|_{1,\omega_T}.
\end{aligned}
$$

Seminorm stability also gives
$|v-J_hv|_{1,T}\leq C|v|_{1,\omega_T}$.
Apply the scaled trace inequality to $w=v-J_hv$ on $T(F)$:

$$
\|w\|_{0,F}^2\leq
C\bigl(h_T^{-1}\|w\|_{0,T}^2+h_T|w|_{1,T}^2\bigr)
\leq Ch_T|v|_{1,\omega_T}^2.
$$

Here $T=T(F)$ and $h_T\simeq h_F$. This proves the edge bound.
For overlap, each cell patch has uniformly bounded overlap and each triangle
can be the selected $T(F)$ for at most three edges. $\square$

The choice $\omega_F=\omega_{T(F)}$ ensures that the patch contains every
moment needed on the chosen cell. A smaller edge neighborhood requires a
separate verification.

### 2.4 The higher-regularity approximation bound

For $v\in H^2(\omega_T)$, choose an affine polynomial $q$ such that

$$
\|v-q\|_{0,\omega_T}+h_T|v-q|_{1,\omega_T}
\leq Ch_T^2|v|_{2,\omega_T}.
$$

This patch estimate follows by applying patch Poincaré to each component
of $\nabla v$, then to $v-b\cdot x$ for the resulting constant vector $b$;
see [Lecture 7, §6.2](lecture-07.md). Since $J_hq=q$, apply
{eq}`eq:lecture-08-preliminary-sz` to $v-q$ and use the triangle inequality:

$$
\|v-J_hv\|_{0,T}+h_T|v-J_hv|_{1,T}
\leq Ch_T^2|v|_{2,\omega_T}.
$$

This completes the integer-order $P_1$ approximation picture. The reliability
argument below uses only the bounds for $v\in H^1$, and therefore does not
assume $H^2$ regularity of the exact Poisson solution.

## 3. A posteriori analysis starts from computable data

An a priori estimate such as
$|u-u_h|_1\leq Ch|u|_2$ predicts approximation behavior under additional
regularity, but its right-hand side contains the unknown exact solution.
An a posteriori estimator is computed from the discrete solution, the problem
data and the mesh. Its local contributions can guide refinement.

We first study

$$
-\Delta u=f\quad\text{in }\Omega,\qquad u=0\quad\text{on }\partial\Omega,
\qquad f\in L^2(\Omega).
$$

The weak solution $u\in H_0^1(\Omega)$ and its conforming Galerkin
approximation $u_h\in V_h$ satisfy

$$
a(u,v)=(f,v)\quad\forall v\in H_0^1(\Omega),\qquad
a(u_h,v_h)=(f,v_h)\quad\forall v_h\in V_h,
$$

where $a(w,v)=\int_\Omega\nabla w\cdot\nabla v$.
Assume exact evaluation of these variational forms and an exact discrete
solve. Quadrature and algebraic errors require additional
terms. No $H^2$ regularity of $u$ is assumed; reentrant corners are allowed.

### 3.1 The dual norm of the residual controls the energy error

Define $R\in(H_0^1(\Omega))'$ by

```{math}
:label: eq:lecture-08-residual-functional
R(v)=(f,v)-a(u_h,v)=a(u-u_h,v).
```

Equivalently, $R=-\Delta(u-u_h)=f+\Delta u_h$ in $H^{-1}(\Omega)$,
where the Laplacians in this identity are distributional.
Equip $H_0^1(\Omega)$ with the full norm $\|\cdot\|_1$ and write

$$
\|R\|_{-1}=\sup_{0\ne v\in H_0^1(\Omega)}\frac{|R(v)|}{\|v\|_1}.
$$

For $e=u-u_h$, testing with $e$ and using Poincaré gives
$|e|_1^2=R(e)\leq\|R\|_{-1}\|e\|_1\leq C_\Omega\|R\|_{-1}|e|_1$.
Consequently $|e|_1\leq C_\Omega\|R\|_{-1}$, including the case $e=0$.
The supremum is over the global test space. To obtain a computable bound,
we first write the action of $R$ as a sum of integrals over cells and edges.

### 3.2 Integration by parts produces cell and edge residuals

The polynomial $u_h|_T$ is smooth on each cell. For $w\in H_0^1(\Omega)$,
cellwise integration by parts, with traces on the edges, gives

$$
R(w)=\sum_T\left[\int_T(f+\Delta u_h)w
                 -\int_{\partial T}(\nabla u_h|_T\cdot n_T)w\right].
$$

This integration differentiates only the discrete polynomial, and requires
no additional regularity of $u$.

For $F=\partial T^+\cap\partial T^-$, let $n^\pm$ be the outward normals of
$T^\pm$. Thus $n^-=-n^+$. Define the cell residual and the scalar normal jump

$$
r_T=f+\Delta(u_h|_T),\qquad
j_F=\nabla u_h^+\cdot n^++\nabla u_h^-\cdot n^-.
$$

Exchanging the labels $+$ and $-$ leaves $j_F$ unchanged. Equivalently,
$j_F=(\nabla u_h^+-\nabla u_h^-)\cdot n^+$.
The trace of $w$ agrees from both sides of $F$, and it vanishes on the exterior
boundary. Summing once over each interior edge gives

```{math}
:label: eq:lecture-08-local-residual
R(w)=\sum_T\int_T r_Tw-\sum_{F\in\mathcal F_h^i}\int_F j_Fw.
```

```{figure} ../assets/lecture-08/residual-normal-jump.svg
:alt: Two adjacent triangles with opposite outward normals on their common edge, illustrating the sum of outward normal derivatives.
:name: fig:lecture-08-residual-jump

The two cell boundary terms combine into $-\int_Fj_Fw$.
Continuity of $u_h$ does not imply continuity of its normal derivative.
```

For affine $P_1$ triangles, $\Delta(u_h|_T)=0$, so $r_T=f|_T$.
The cell residual can therefore remain nonzero even after the Galerkin
linear system has been solved exactly. The vector residual of that linear
system records the action of $R$ on the basis of $V_h$.

## 4. Scott–Zhang localizes the dual residual norm

**Step 1. Use Galerkin orthogonality in the dual norm.**
The discrete equation gives $R(v_h)=0$ for all $v_h\in V_h$. Thus
$R(v)=R(v-v_h)$ for every $v\in H_0^1(\Omega)$ and every $v_h\in V_h$.
In the supremum defining $\|R\|_{-1}$, we may choose $v_h$ as a function
of the test function $v$.

**Step 2. Integrate by parts and group the residual by cells.**
Set $w=v-v_h\in H_0^1(\Omega)$ and let $\mathcal F_T^i$ be the set of
interior edges of $T$. Apply the cellwise integration by parts of Section 3.2
to $w$. Its trace is zero on $\partial\Omega$ and single-valued on each
interior edge. Grouping {eq}`eq:lecture-08-local-residual` by cells assigns
half of each jump integral to each adjacent cell.

$$
R(v-v_h)=\sum_{T\in\mathcal T_h}\left(
  \int_T r_T w-\frac12\sum_{F\in\mathcal F_T^i}\int_F j_F w\right).
$$

Cauchy–Schwarz on each cell and edge gives

$$
|R(v-v_h)|\leq\sum_{T\in\mathcal T_h}\left(
  \|r_T\|_{0,T}\|w\|_{0,T}
 +\frac12\sum_{F\in\mathcal F_T^i}\|j_F\|_{0,F}\|w\|_{0,F}\right).
$$

**Step 3. Choose the Scott–Zhang interpolant.**
Set $v_h=J_hv$. Boundary preservation gives $J_hv\in V_h$, so

```{math}
:label: eq:lecture-08-residual-sz
R(v)=R(v-J_hv).
```

By {prf:ref}`thm:lecture-08-sz-residual`,

$$
\|w\|_{0,T}\leq Ch_T|v|_{1,\omega_T},
\qquad
\|w\|_{0,F}\leq Ch_F^{1/2}|v|_{1,\omega_F}.
$$

For each cell define
$c_T(v)=\max\bigl(\{|v|_{1,\omega_T}\}\cup
\{|v|_{1,\omega_F}:F\in\mathcal F_T^i\}\bigr)$.
Inserting the interpolation bounds into Step 2 yields

$$
|R(v)|\leq C\sum_{T\in\mathcal T_h}c_T(v)\left(
h_T\|r_T\|_{0,T}
+\frac12\sum_{F\in\mathcal F_T^i}h_F^{1/2}\|j_F\|_{0,F}\right).
$$

**Step 4. Bound the patch overlap and take the supremum.**
Bounding each squared maximum by the sum of its squared terms gives

$$
\sum_{T\in\mathcal T_h}c_T(v)^2
\leq \sum_{T\in\mathcal T_h}|v|_{1,\omega_T}^2
+2\sum_{F\in\mathcal F_h^i}|v|_{1,\omega_F}^2
\leq C|v|_1^2.
$$

Each interior edge occurs twice in the sum over cells. For the last
inequality, expand each patch seminorm squared as the sum of
$\int_K|\nabla v|^2$ over its cells $K$. Each $K$ belongs to a uniformly
bounded number of cell patches by the vertex-star bound. It also belongs
to a uniformly bounded number of edge patches, since
$\omega_F=\omega_{T(F)}$ and at most three edges can select the same
triangle $T(F)$.

Use weighted Cauchy–Schwarz in the parenthesis of Step 3, with weight $1$
for the cell term and $1/2$ for each edge term. The sum of the weights is
uniformly bounded because a triangle has at most three interior edges.
Cauchy–Schwarz over the cells and the overlap bound then give

$$
|R(v)|\leq C |v|_1
\left[
\sum_{T\in\mathcal T_h}\left(
h_T^2\|r_T\|_{0,T}^2
+\frac12\sum_{F\in\mathcal F_T^i} h_F\|j_F\|_{0,F}^2
\right)
\right]^{1/2}.
$$

Since $|v|_1\leq\|v\|_1$, taking the supremum in the definition of
$\|R\|_{-1}$ yields the bound

```{math}
:label: eq:lecture-08-dual-residual-bound
\|R\|_{-1}\leq C
\left[
\sum_{T\in\mathcal T_h} h_T^2\|r_T\|_{0,T}^2
+\sum_{F\in\mathcal F_h^i} h_F\|j_F\|_{0,F}^2
\right]^{1/2}.
```

Here each interior edge is counted once. The constant depends on the
shape-regularity bound and is independent of the mesh sizes and the data.

## 5. The dual residual bound defines the local indicator and global estimator

The right-hand side of {eq}`eq:lecture-08-dual-residual-bound` uses only
$f$, $u_h$ and the mesh. The interpolation bounds give the weights
$h_T$ and $h_F^{1/2}$ before squaring. Assign the cell contribution to its
cell and half of each squared jump contribution to each adjacent cell.
This defines the local indicator $e_T$ and the global estimator $\eta$.

```{math}
:label: eq:lecture-08-estimator
e_T^2=h_T^2\|r_T\|_{0,T}^2
+\frac12\sum_{F\in\mathcal F_T^i}h_F\|j_F\|_{0,F}^2,
\qquad
\eta=\left(\sum_{T\in\mathcal T_h}e_T^2\right)^{1/2},
\qquad e_T\geq 0.
```

Every interior edge belongs to exactly two cells, so $\eta$ is the square
root in {eq}`eq:lecture-08-dual-residual-bound` and $\|R\|_{-1}\leq C\eta$.
The individual $e_T$ can guide local refinement.

```{prf:theorem} Global reliability for the conforming Poisson estimator
:label: thm:lecture-08-reliability

Under the standing assumptions and the variational assumptions of Section 3,

$$
|u-u_h|_1\leq C_{\mathrm{rel}}\eta.
$$

The constant depends on the shape-regularity bound and is independent of the
mesh sizes, the data and the solution. The full error satisfies
$\|u-u_h\|_1\leq C_\Omega C_{\mathrm{rel}}\eta$ by Poincaré.
```

The dual norm bound and Section 3.1 already give $|u-u_h|_1\leq C_\Omega C\eta$.
To obtain a constant depending only on shape regularity, use the estimate
for $|R(v)|$ before taking the supremum. With $v=e=u-u_h$ it gives $|e|_1^2=R(e)\leq C\eta |e|_1$. This proves
reliability, with the case $e=0$ immediate. The argument applies to weak
solutions in $H_0^1(\Omega)$ on shape-regular, possibly graded meshes.

For general $f\in L^2(\Omega)$, evaluating the cell norms can require
numerical integration. The bound uses the exact norms. If $f$ is piecewise
polynomial, sufficiently accurate quadrature evaluates them exactly.

## 6. Examples, exercises and the next step

1. **Dual edge moments.** Use the biorthogonality in Section 1.2 to explain
   how the moment for a vertex of a $P_1$ edge recovers its nodal coefficient.
   Explain why using the ordinary edge average for both vertices fails to
   recover the two coefficients.
2. **Choice of averaging set.** In {numref}`fig:lecture-08-sz-all-dofs`, locate
   an interior vertex, a boundary vertex, an edge-interior node and a
   cell-interior node. Identify $\Xi_i$ in each case. Give a second admissible
   choice for the interior vertex and explain the boundary restriction.
3. **Patch locality.** In {numref}`fig:lecture-08-sz-patch`, identify which
   selected edges lie outside $T$. Explain why data on $T$ alone do not
   determine $J_hv|_T$.
4. **A resolved linear system still has a residual estimator.** On a uniform
   square grid split into triangles, take $f=1$ and $P_1$ elements. Show that
   the cell part of $e_T^2$ is $h_T^2|T|$, regardless of the algebraic
   solver tolerance. Explain why this does not contradict $R(v_h)=0$.
5. **The normal jump is necessary.** Consider two triangles sharing a
   horizontal edge. Let a continuous piecewise affine function be zero in
   the upper triangle and equal to $y$ in the lower triangle. Compute the
   two outward normal derivatives on the common edge and $j_F$. Its broken
   Laplacian vanishes in both cells. This is a local example; no homogeneous
   exterior boundary condition is imposed on this two-cell patch.
6. **Face bookkeeping.** Verify that $\sum_T e_T^2$ equals the sum inside
   the square root in {eq}`eq:lecture-08-dual-residual-bound`. Explain the
   factor $1/2$ in the definition of $e_T$.

The next theoretical lecture will study the reverse direction: control of
an indicator by the energy error on a patch, with a data-oscillation term.
It will introduce cell and edge bubble functions and distinguish local
efficiency from global reliability. If $f$ is replaced by a polynomial
projection $f_h$ in the estimator, the omitted data residual must also be
controlled. No efficiency estimate is claimed in this lecture.

**Further reading.** The moment construction is due to
{cite:ts}`ScottZhang1990`. Background on local approximation and residual
estimation is available in {cite:ts}`BrennerScott2010` and
{cite:ts}`ErnGuermond2004`.
