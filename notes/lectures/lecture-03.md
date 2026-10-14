# Lecture 3 — Polynomial approximation and a priori error estimates

The main arguments developed in this lecture are

```{math}
\text{Deny–Lions}\ \longrightarrow\ \text{Bramble–Hilbert}
\ \longrightarrow\ \text{local interpolation estimates}
\ \longrightarrow\ \text{Céa's lemma}.
```

## Objectives and prerequisites

By the end of the lecture, you should be able to prove the equivalence between
a Sobolev seminorm and a quotient norm modulo polynomials, estimate an operator
that annihilates polynomials, transfer reference-cell estimates to a physical
mesh, and derive an energy-norm error bound for a conforming Galerkin method.

We use Sobolev spaces, affine scaling and shape regularity from
[Lecture 2](lecture-02.md), and {prf:ref}`thm:lecture-01-cea` from
[Lecture 1](lecture-01.md). We recall the precise compactness and embedding
results used below; their proofs belong to the theory of Sobolev spaces.
The construction and unisolvence of general Lagrange elements will be developed
in Lecture 4. Here their required properties are stated explicitly.

Throughout, $m\geq1$ is an integer, $1\leq p<\infty$, and
$\mathbb P^{m-1}(D)$ denotes polynomials of **total degree** at most $m-1$
restricted to $D$. Constants may depend on $m,p,d$ and the fixed domain;
mesh-independent estimates will state their geometric assumptions.

## 1. The kernel of the highest-order seminorm

On a bounded, connected Lipschitz domain $D\subset\mathbb R^d$, the seminorm

```{math}
|v|_{W^{m,p}(D)}
=\left(\sum_{|\alpha|=m}\|D^\alpha v\|_{L^p(D)}^p\right)^{1/p}
```

vanishes precisely on $\mathbb P^{m-1}(D)$.

To justify the nontrivial implication, first recall that a distribution with
zero gradient on a connected open set is constant. Locally this follows by
mollification on balls: its mollifications have zero gradient and are constant
on smaller balls. Passing to the distributional limit and using overlapping
balls gives the same constant throughout the connected set.

If all order-$m$ derivatives of $v$ vanish, every derivative $D^\beta v$ with
$|\beta|=m-1$ is therefore a constant $c_\beta$. The polynomial

```{math}
q_{m-1}(x)=\sum_{|\beta|=m-1}\frac{c_\beta}{\beta!}x^\beta
```

has exactly those order-$(m-1)$ derivatives. Subtract it and apply induction
to $v-q_{m-1}$. This gives a polynomial of degree at most $m-1$ representing
$v$ almost everywhere. The case $m=1$ is the zero-gradient assertion.

Connectedness matters: on a disconnected domain, the kernel contains functions
that are independently polynomial on each connected component.

## 2. The Deny–Lions lemma

````{prf:theorem} Deny–Lions
:label: thm:lecture-03-deny-lions

Let $D$ be a bounded, connected Lipschitz domain. For every integer $m\geq1$
and $1\leq p<\infty$, there exists $C_D>0$ such that

```{math}
:label: eq:lecture-03-quotient
|v|_{W^{m,p}(D)}
\leq \inf_{q\in\mathbb P^{m-1}(D)}\|v-q\|_{W^{m,p}(D)}
\leq C_D|v|_{W^{m,p}(D)}
\qquad\forall v\in W^{m,p}(D).
```

Thus the seminorm induces a norm equivalent to the quotient norm on
$W^{m,p}(D)/\mathbb P^{m-1}(D)$.
````

**Proof.** Set $X=W^{m,p}(D)$ and $Q=\mathbb P^{m-1}(D)$.
The first inequality follows from $|v-q|_{W^{m,p}}=|v|_{W^{m,p}}$.
For the other inequality, choose a basis $q_1,\ldots,q_N\in Q$ and its
**dual basis** $q^1,\ldots,q^N\in Q'$, characterized by
$q^i(q_j)=\delta_{ij}$. Here $Q$ carries the norm inherited from $X$;
finite dimensionality makes every $q^i$ continuous.

The inclusion $j:Q\hookrightarrow X$ induces the bounded **restriction map**

```{math}
j':X'\longrightarrow Q',\qquad j'\lambda=\lambda|_Q.
```

Hahn–Banach makes $j'$ surjective: every functional on $Q$ admits a continuous
extension to $X$. There is no canonical inclusion of $Q'$ into $X'$ from
$j$ alone, because these extensions are generally not unique.
Choose extensions $\lambda_i\in X'$ of the dual basis functionals, so that
$j'\lambda_i=q^i$ and $\lambda_i(q_j)=\delta_{ij}$. These choices define a
bounded linear map

```{math}
E:Q'\longrightarrow X',\qquad
E\mu=\sum_{i=1}^N\mu(q_i)\lambda_i,\qquad j'E=\mathrm{Id}_{Q'}.
```

Thus $E$ identifies $Q'$ with a subspace of $X'$ after a choice of extensions;
this identification is noncanonical. Define the bounded projection

```{math}
R v:=\sum_{i=1}^N\lambda_i(v)q_i,\qquad Rq=q\quad(q\in Q).
```

We prove

```{math}
:label: eq:lecture-03-norm-equivalence
\|v\|_X\leq C\left(|v|_{W^{m,p}(D)}^p+
\sum_{i=1}^N|\lambda_i(v)|^p\right)^{1/p}.
```

If this failed, normalization would give a sequence $v_n$ with $\|v_n\|_X=1$
and both $|v_n|_{W^{m,p}}\to0$ and $\lambda_i(v_n)\to0$.
The compact embedding
$W^{m,p}(D)\hookrightarrow W^{m-1,p}(D)$ provides a subsequence converging
strongly in $W^{m-1,p}(D)$. Moreover,

```{math}
\|v_n-v_j\|_X^p
=\|v_n-v_j\|_{W^{m-1,p}(D)}^p+|v_n-v_j|_{W^{m,p}(D)}^p,
\qquad
|v_n-v_j|_{W^{m,p}}\leq |v_n|_{W^{m,p}}+|v_j|_{W^{m,p}}.
```

Consequently, this subsequence is Cauchy in $X$ and converges strongly in $X$
to some $v$. Continuity gives $\|v\|_X=1$, $|v|_{W^{m,p}}=0$ and
$\lambda_i(v)=0$ for every $i$. The kernel characterization implies $v\in Q$;
its zero coordinates imply $v=0$, a contradiction.

Apply {eq}`eq:lecture-03-norm-equivalence` to $v-Rv$. All its coordinates
vanish, and its order-$m$ derivatives equal those of $v$. Hence

```{math}
\inf_{q\in Q}\|v-q\|_X\leq\|v-Rv\|_X\leq C|v|_{W^{m,p}(D)}.
```

This proves the lemma. The compact embedding used here holds for the stated
bounded Lipschitz domains and finite $p$, including $p=1$; no weak subsequence
argument or reflexivity is needed. See {cite:ts}`BrennerScott2010` for the
Sobolev-space framework.

The projection $R$ is an auxiliary device for this proof. It does not prescribe
nodal values or ensure matching traces across a mesh.

````{prf:remark} Deny–Lions as a generalized Poincaré inequality
:label: rem:lecture-03-generalized-poincare

Under the assumptions of the lemma, let $P:X\to Q$ be any bounded linear
projection, where $X=W^{m,p}(D)$ and $Q=\mathbb P^{m-1}(D)$. Since
$(\mathrm{Id}-P)q=0$ for $q\in Q$, Deny–Lions gives

```{math}
\|v-Pv\|_{W^{m,p}(D)}
\leq \|\mathrm{Id}-P\|_{\mathcal L(X,X)}
\inf_{q\in Q}\|v-q\|_{W^{m,p}(D)}
\leq C|v|_{W^{m,p}(D)}.
```

In particular, on $\ker P$ the full norm is controlled by the highest-order
seminorm. The constant may depend on the chosen projection.

A concrete choice is the moment projection $P_Dv\in Q$ defined by

```{math}
\int_D(v-P_Dv)q\,\mathrm dx=0\qquad\forall q\in Q.
```

Its coefficient matrix in a polynomial basis is the positive-definite Gram
matrix $G_{ij}=\int_Dq_iq_j$. The moments are bounded functionals on $L^p(D)$,
including $p=1$, since the polynomials are bounded on $D$. Thus $P_D$ is
well-defined, bounded on $X$, and reproduces $Q$. Consequently,

```{math}
\|v\|_{W^{m,p}(D)}\leq C|v|_{W^{m,p}(D)}
\quad\text{whenever}\quad
\int_D vq\,\mathrm dx=0\quad\forall q\in\mathbb P^{m-1}(D).
```

For $m=1$, $P_Dv$ is the mean $v_D=|D|^{-1}\int_Dv$, and this becomes

```{math}
\|v-v_D\|_{W^{1,p}(D)}\leq C\|\nabla v\|_{L^p(D)}.
```

This is the mean-value form of Poincaré's inequality. For $m=2$, removing
the affine kernel requires the moments against $1,x_1,\ldots,x_d$ to vanish;
zero mean alone does not control nonconstant affine functions by their
second derivatives.
````

## 3. Bramble–Hilbert for operators that annihilate polynomials

````{prf:theorem} Bramble–Hilbert, operator form
:label: thm:lecture-03-bramble-hilbert

Under the assumptions of {prf:ref}`thm:lecture-03-deny-lions`, let $Y$ be a
Banach space and let $\tau:W^{m,p}(D)\to Y$ be bounded and linear. If
$\tau q=0$ for every $q\in\mathbb P^{m-1}(D)$, then

```{math}
\|\tau v\|_Y\leq C_D\|\tau\|_{\mathcal L(W^{m,p}(D),Y)}
|v|_{W^{m,p}(D)}.
```
````

**Proof.** For every $q\in\mathbb P^{m-1}(D)$, linearity gives
$\tau v=\tau(v-q)$. Bound this by $\|\tau\|\|v-q\|_{W^{m,p}}$, take the
infimum over $q$, and apply Deny–Lions.

In particular, if a bounded linear operator $I$ reproduces these polynomials,
then $\tau=\mathrm{Id}-I$ satisfies the hypothesis. Both **boundedness on the
chosen Sobolev space** and **polynomial reproduction** must be checked.
Reproduction alone does not provide an error estimate.

## 4. Reference-cell interpolation

Fix a reference simplex $\widehat T$ and a polynomial degree $k\geq1$.
For an integer $1\leq m\leq k+1$, assume that a linear reference operator
$\widehat I_k$ has the following properties:

1. $\widehat I_k:W^{m,p}(\widehat T)\to\mathbb P^k(\widehat T)$ is bounded;
2. $\widehat I_k q=q$ for every $q\in\mathbb P^k(\widehat T)$.

Since polynomial norms are equivalent on this fixed finite-dimensional space,
$\widehat I_k$ is bounded into $W^{s,p}(\widehat T)$ for every integer
$0\leq s\leq m$. Bramble–Hilbert applied to
$\mathrm{Id}-\widehat I_k$ therefore gives

```{math}
:label: eq:lecture-03-reference-estimate
\|\widehat v-\widehat I_k\widehat v\|_{W^{s,p}(\widehat T)}
\leq C_{k,m,p,d}|\widehat v|_{W^{m,p}(\widehat T)},
\qquad 0\leq s\leq m\leq k+1.
```

For nodal interpolation, these assumptions hold if the nodes are unisolvent
for $\mathbb P^k$ and $mp>d$. Indeed, Sobolev embedding gives a continuous
representative and bounds every nodal evaluation by
$C\|\widehat v\|_{W^{m,p}}$. Writing
$\widehat I_k\widehat v=\sum_i\widehat v(\widehat a_i)\widehat\varphi_i$
then proves boundedness; unisolvence proves reproduction.

We will establish unisolvence for the standard Lagrange nodes in Lecture 4.
The condition $mp>d$ is a sufficient condition used throughout the nodal
argument here. In particular, arbitrary $H^1$ functions in two dimensions
do not have well-defined nodal values. Interpolants for low-regularity functions
will be introduced later.

## 5. Local estimates and shape regularity

Let $\mathcal T_h$ be a conforming simplicial mesh of a polygonal or polyhedral
domain, with affine maps $F_T(\widehat x)=B_T\widehat x+b_T$.
Set $h_T=\operatorname{diam}(T)$, let $\rho_T$ be its inradius, and let
$J_T=|\det B_T|$. Define the transported operator by

```{math}
(I_Tv)\circ F_T=\widehat I_k(v\circ F_T).
```

This definition, and not merely the fact that $I_Tv$ is polynomial, ensures
that interpolation commutes with pullback. For $e_T=v-I_Tv$, reference-cell
scaling and {eq}`eq:lecture-03-reference-estimate` give

```{math}
\begin{aligned}
|e_T|_{W^{s,p}(T)}
&\leq C\|B_T^{-1}\|^s J_T^{1/p}
|e_T\circ F_T|_{W^{s,p}(\widehat T)}\\
&\leq C\|B_T^{-1}\|^s J_T^{1/p}
|v\circ F_T|_{W^{m,p}(\widehat T)}\\
&\leq C\|B_T^{-1}\|^s\|B_T\|^m |v|_{W^{m,p}(T)}.
\end{aligned}
```

Here the order-zero seminorm means the $L^p$ norm. The determinant factors
cancel. Since $\|B_T\|\leq Ch_T$ and
$\|B_T^{-1}\|\leq C\rho_T^{-1}$, this yields
$|e_T|_{W^{s,p}(T)}\leq Ch_T^m\rho_T^{-s}|v|_{W^{m,p}(T)}$.
For a shape-regular family, $h_T/\rho_T\leq\sigma^{-1}$ uniformly, so

```{math}
:label: eq:lecture-03-local-estimate
|v-I_Tv|_{W^{s,p}(T)}
\leq C h_T^{m-s}|v|_{W^{m,p}(T)},\qquad 0\leq s\leq m.
```

The constant depends on $k,m,p,d$, the reference element, its interpolation
operator and $\sigma$, but not on $T$ or its size. We keep $k$ fixed; this
argument gives no uniform estimate as $k\to\infty$.

For $h_T\leq1$, applying this estimate at every derivative order $j\leq s$
also gives
$\|v-I_Tv\|_{W^{s,p}(T)}\leq Ch_T^{m-s}|v|_{W^{m,p}(T)}$.
This follows from $h_T^{m-j}\leq h_T^{m-s}$. It does not require lower-order
derivatives of $v$ to be controlled by its highest-order seminorm.

## 6. From local approximation to a conforming function

Local estimates initially control a **broken** norm. To use Céa's lemma we
also need a global interpolant $I_hv\in V_h\subset H_0^1(\Omega)$.
Assume that the local operators use matching Lagrange nodes on common faces,
that those face nodes are unisolvent for the trace polynomial space, and that
boundary nodes prescribe homogeneous Dirichlet data. For a continuous $v$
with zero boundary values, the resulting $I_hv$ has matching traces and zero
trace on $\partial\Omega$.

For $P_1$ triangles this can be checked immediately: the affine interpolant
is determined by the three vertex values; two neighboring triangles give
linear edge traces with the same two endpoint values. Those traces agree
on the entire edge. A boundary edge has two zero endpoint values and hence
zero trace. For higher degrees we use the analogous face-unisolvence property,
whose construction is deferred to Lecture 4.

Matching traces imply $I_hv\in H^1(\Omega)$: integration by parts cell by
cell against a smooth compactly supported test function cancels the interior
face terms, leaving the piecewise gradient as the distributional gradient.
It lies in $L^2$, and zero boundary trace gives $I_hv\in H_0^1$.
Independent polynomial projections on each cell generally lack this property.

## 7. The energy-norm error bound

````{prf:theorem} A priori estimate for conforming Galerkin approximation
:label: thm:lecture-03-energy-error

Let $\Omega\subset\mathbb R^2$ be a bounded polygonal domain with a
shape-regular family of conforming triangular meshes, $h=\max_T h_T\leq1$.
Let $V=H_0^1(\Omega)$ with norm $|\cdot|_{H^1(\Omega)}$, and let $a$ be
bounded and coercive on $V$, with constants $M$ and $\alpha$.

Assume a conforming degree-$k$ space $V_h\subset V$ with the nodal
interpolant described above. If the weak solution satisfies
$u\in H^m(\Omega)\cap V$ for an integer $2\leq m\leq k+1$, its Galerkin
approximation satisfies

```{math}
:label: eq:lecture-03-energy-local
|u-u_h|_{H^1(\Omega)}
\leq C\frac{M}{\alpha}
\left(\sum_{T\in\mathcal T_h}h_T^{2(m-1)}|u|_{H^m(T)}^2\right)^{1/2}
\leq C\frac{M}{\alpha}h^{m-1}|u|_{H^m(\Omega)}.
```

The constant is independent of mesh sizes and depends on the fixed degree,
the reference interpolation operator and the shape-regularity bound.
````

**Proof.** Here $m>d/2$, so nodal evaluation is bounded and $u$ has a
continuous representative. Its zero trace gives zero boundary values.
Consequently, $I_hu\in V_h$. Céa and the local estimate with $s=1$, $p=2$
give

```{math}
\begin{aligned}
|u-u_h|_{H^1(\Omega)}
&\leq\frac{M}{\alpha}|u-I_hu|_{H^1(\Omega)}\\
&=\frac{M}{\alpha}\left(\sum_T|u-I_Tu|_{H^1(T)}^2\right)^{1/2}\\
&\leq C\frac{M}{\alpha}
\left(\sum_T h_T^{2(m-1)}|u|_{H^m(T)}^2\right)^{1/2}.
\end{aligned}
```

Since $h_T\leq h$ and the cells partition $\Omega$, the global bound follows.
Poincaré's inequality gives the corresponding full $H^1$-norm bound.
No comparison between the smallest and largest cell sizes was used:
**shape regularity suffices; quasi-uniformity is unnecessary for this estimate.**

### Poisson and solution regularity

For $a(w,v)=\int_\Omega\nabla w\cdot\nabla v$, the energy norm chosen above
gives $M=\alpha=1$. Thus, for $P_1$ triangles and $u\in H^2\cap H_0^1$,

```{math}
|u-u_h|_{H^1(\Omega)}\leq Ch|u|_{H^2(\Omega)}.
```

If $\Omega$ is a bounded convex polygon and $f\in L^2(\Omega)$, elliptic
regularity gives $u\in H^2(\Omega)$ and
$\|u\|_{H^2}\leq C_\Omega\|f\|_{L^2}$. Invoking this separate PDE result
then yields $|u-u_h|_{H^1}\leq CC_\Omega h\|f\|_{L^2}$; see
{cite:ts}`BrennerScott2010` and {cite:ts}`ErnGuermond2004` for regularity and
finite element error estimates. Convexity does not provide arbitrary higher
regularity at polygonal corners.

For degree $k$, the rate is $h^{m-1}$ at the available integer regularity
$2\leq m\leq k+1$. Increasing $k$ improves the proven rate only when the
solution has the additional regularity. A general Lipschitz domain with
$f\in L^2$ need not give $u\in H^2$; the present nodal proof does not cover
all low-regularity solutions. Fractional regularity and appropriate
quasi-interpolants require further analysis.

The estimate $\|u-I_hu\|_{L^2}\leq Ch^m|u|_{H^m}$ concerns **interpolation**.
Céa in the energy norm does not imply the same $L^2$ rate for $u-u_h$.
That additional Galerkin estimate will use duality in Lecture 5.

## 8. Exercises

1. **Poincaré modulo constants.** For $m=1$, use the bounded functional
   $\lambda(v)=|D|^{-1}\int_D v$ in the Deny–Lions proof and deduce
   $\|v-\lambda(v)\|_{W^{1,p}(D)}\leq C|v|_{W^{1,p}(D)}$.
2. **Local-to-global estimate.** For $P_1$ triangles, derive
   $|u-u_h|_{H^1}^2\leq C\sum_T h_T^2|u|_{H^2(T)}^2$.
   Identify the assumptions that guarantee $I_hu\in H_0^1$ and explain why
   replacing each local interpolant with an independent $L^2$ projection
   does not justify this proof.
3. **Scaling check, independent study.** On $(0,h)$, interpolate $v(x)=x^2$
   at the endpoints. Compute exactly
   $\|v-I_1v\|_{L^2}=h^{5/2}/\sqrt{30}$ and
   $|v-I_1v|_{H^1}=h^{3/2}/\sqrt3$, and compare them with
   $h^2|v|_{H^2}$ and $h|v|_{H^2}$. Here $|v|_{H^2}=2\sqrt h$.

## Further reading

Polynomial approximation, reference-element interpolation and conforming error
estimates are treated in {cite:ts}`Ciarlet1978`,
{cite:ts}`BrennerScott2010` and {cite:ts}`ErnGuermond2004`.
The central proofs above are complete; the compact Sobolev embedding, the
embedding into continuous functions and the convex-polygon elliptic regularity
result are invoked with explicit assumptions.
