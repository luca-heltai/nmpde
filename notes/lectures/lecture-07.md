# Lecture 7 — Traces, inverse estimates, and Scott–Zhang interpolation

## Objectives and prerequisites

This lecture consolidates the passage from the continuous boundary-value problem
to global finite element error estimates. By the end, you should be able to

- formulate nonhomogeneous Dirichlet conditions using traces and a lifting;
- derive inverse estimates with separate derivative and integrability indices,
  then apply them to polynomial traces;
- distinguish stability of the continuous problem, approximation, and Galerkin error;
- construct the Scott–Zhang operator for linear triangles and prove its projection,
  boundary, stability, and approximation properties.

We use affine scaling from [Lecture 2](lecture-02.md), polynomial approximation
and Céa from [Lecture 3](lecture-03.md), conforming Lagrange spaces from
[Lecture 4](lecture-04.md), and duality from [Lecture 6](lecture-06.md).
The residual estimator and its reliability and efficiency analysis are deferred.

**Standing assumptions.** For the continuous problem, $\Omega\subset\mathbb R^2$
is a bounded, connected Lipschitz domain. For the finite element analysis it is
a polygon with a family of conforming, affine triangular meshes $\mathcal T_h$,
without hanging nodes. In Sections 2–3 the cells are nondegenerate, with no
uniform shape assumption: the factors $h_T$, $\rho_T$, and $J_T$ remain
explicit. From Section 4 onward, assume the family is shape regular,
$\sup_{h,T}h_T/\rho_T<\infty$. Set $h=\max_T h_T\leq1$ and keep polynomial
degrees fixed. Constants are independent of mesh sizes; in Sections 2–3 they
are also independent of cell distortion, unless a shape-regular corollary is
explicitly invoked. From Section 4 they may depend on the shape-regularity
bound. No global quasi-uniformity is assumed unless stated.

We write $\|v\|_{s,D}$ for the full Sobolev norm of order $s$ on $D$ and
$|v|_{s,D}$ for its seminorm; in particular, $\|v\|_{0,D}^2=\int_D|v|^2$
and $|v|_{1,D}=\|\nabla v\|_{0,D}$. An omitted domain means $D=\Omega$.
The index $-1$ denotes the dual norm on $(H_0^1(\Omega))'$.

## 1. Using traces to define Dirichlet conditions

An $H^1(\Omega)$ function is an equivalence class modulo sets of area zero.
Its values on $\partial\Omega$ cannot be obtained by selecting an arbitrary
representative: changing values on the boundary changes no interior norm.
In two dimensions, $H^1$ also has no continuous embedding into $C^0$.

```{prf:theorem} Trace and lifting
:label: thm:lecture-07-trace

On a bounded Lipschitz domain there is a bounded, surjective linear map

$$
\gamma:H^1(\Omega)\longrightarrow H^{1/2}(\partial\Omega)
$$

agreeing with restriction for functions smooth up to the boundary. Its kernel
is $H_0^1(\Omega)$, and it has a bounded linear right inverse

$$
E:H^{1/2}(\partial\Omega)\longrightarrow H^1(\Omega),
\qquad \gamma Eg=g.
$$

Thus $\|\gamma v\|_{1/2,\partial\Omega}\leq C\|v\|_{1,\Omega}$
and $\|Eg\|_{1,\Omega}\leq C\|g\|_{1/2,\partial\Omega}$.
```

This Sobolev-space theorem is used without proof; see
{cite:ts}`BrennerScott2010` and {cite:ts}`ErnGuermond2004`.
The boundary norm has the equivalent characterization

$$
\|g\|_{1/2,\partial\Omega}\simeq
\inf\{\|G\|_{1,\Omega}:\gamma G=g\}.
$$

On a polygon this norm is defined on the **whole boundary**. Independent
edgewise data need not belong to $H^{1/2}(\partial\Omega)$: jumps at vertices can violate its fractional regularity.
The lifting $E$ is a continuous function-space operator; choosing it does not
specify an algorithm for a discrete boundary extension.

### 1.1 A lifting gives an affine trial space

Consider full Dirichlet conditions for Poisson:

$$
-\Delta u=f\quad\text{in }\Omega,\qquad \gamma u=g\quad\text{on }\partial\Omega,
$$

with $f\in H^{-1}(\Omega)=(H_0^1(\Omega))'$ and
$g\in H^{1/2}(\partial\Omega)$. Write

$$
V=H_0^1(\Omega),\qquad V_g=Eg+V=\{v\in H^1(\Omega):\gamma v=g\}.
$$

The weak problem is: find $u\in V_g$ such that

$$
a(u,v)=\langle f,v\rangle\quad\forall v\in V,
\qquad a(w,v)=\int_\Omega\nabla w\cdot\nabla v.
$$

The trial space is affine; the test space is linear. Set $u=Eg+w$ and solve

$$
a(w,v)=\langle f,v\rangle-a(Eg,v)\quad\forall v\in V.
$$

Poincaré makes $|\cdot|_1$ a norm on $V$. The form is continuous and coercive
in this norm, and the right-hand side is bounded by

$$
\bigl(C_P\|f\|_{-1}+|Eg|_1\bigr)|v|_1,
$$

where $H^{-1}$ here uses the full norm $\|\cdot\|_1$ on $V$ and $C_P$ includes the
corresponding norm-equivalence constant. Lax–Milgram gives a unique $w$, hence

```{math}
:label: eq:lecture-07-continuous-stability
\boxed{\|u\|_{1,\Omega}\leq
C_\Omega\bigl(\|f\|_{-1,\Omega}+
\|g\|_{1/2,\partial\Omega}\bigr).}
```

**Independence of the lifting.** Let $G_1,G_2\in H^1(\Omega)$ satisfy
$\gamma G_1=\gamma G_2=g$. By linearity of the trace and the identity
$\ker\gamma=V$, their difference $G_1-G_2$ belongs to $V$. In particular,
$G_1+V=G_2+V$: both liftings describe the same affine trial space.

For $i=1,2$, let $w_i\in V$ be the unique solution of

$$
a(w_i,v)=\langle f,v\rangle-a(G_i,v)\qquad\forall v\in V,
$$

and set $u_i=G_i+w_i$. Subtracting the two equations gives

$$
a(w_1-w_2,v)=-a(G_1-G_2,v)\qquad\forall v\in V.
$$

Consequently, the difference of the reconstructed solutions,

$$
d=u_1-u_2=(G_1-G_2)+(w_1-w_2),
$$

belongs to $V$ and satisfies $a(d,v)=0$ for every $v\in V$. Since $d\in V$,
we may choose $v=d$ and obtain

$$
0=a(d,d)=\int_\Omega|\nabla d|^2=|d|_{1,\Omega}^2.
$$

Poincaré's inequality now yields
$\|d\|_{0,\Omega}\leq C_P|d|_{1,\Omega}=0$, hence $u_1=u_2$ in
$H^1(\Omega)$. Equivalently,

$$
w_2=w_1+G_1-G_2.
$$

Thus changing the lifting changes the homogeneous unknown by exactly the
opposite amount, leaving the solution $u=G_i+w_i$ unchanged. $\square$

This is a stability estimate for the continuous solution. It contains no mesh
and asserts no $H^2$ regularity. Such regularity requires separate assumptions
on the domain and the data, including the boundary data.

## 2. Inverse inequalities

In Sections 2–3, $k\geq0$ denotes the fixed polynomial degree, while $p,q$
denote integrability exponents. The local arguments hold in dimension $d\geq2$
on nondegenerate affine simplices; triangles correspond to $d=2$.
We use the convention $1/\infty=0$.

For integer $m\geq0$, write $\|v\|_{m,p,D}$ and $|v|_{m,p,D}$ for the full
norm and seminorm in $W^{m,p}(D)$. For $m=0$ both mean $\|v\|_{0,p,D}$.
For $p<\infty$,

$$
|v|_{m,p,D}=\left(\sum_{|\alpha|=m}\|D^\alpha v\|_{0,p,D}^p\right)^{1/p},
\qquad
\|v\|_{m,p,D}=\left(\sum_{j=0}^m|v|_{j,p,D}^p\right)^{1/p}.
$$

For $p=\infty$, replace these finite sums by maxima and use essential suprema.
When $p=2$ we continue to omit the integrability index:
$\|v\|_{m,D}=\|v\|_{m,2,D}$ and $|v|_{m,D}=|v|_{m,2,D}$.

### 2.1 Affine geometry separates volume, length, and distortion

Let $F_T(\widehat x)=B_T\widehat x+b_T$ map the fixed reference simplex onto
$T$, and set

$$
J_T=|\det B_T|=\frac{|T|}{|\widehat T|},\qquad
h_T=\operatorname{diam}T,\qquad
\rho_T=\text{radius of the largest ball contained in }T.
$$

The notation $J_T$ denotes the **absolute determinant**, as in Lecture 2.
Write $a_T$ and $b_T^*$ for the smallest and largest singular values of $B_T$;
the superscript distinguishes $b_T^*$ from the translation vector $b_T$.
Then $\|B_T\|=b_T^*$, $\|B_T^{-1}\|=a_T^{-1}$, and, with constants depending
only on the reference simplex,

$$
b_T^*\simeq h_T,\qquad a_T\simeq\rho_T,\qquad
c\rho_T^d\leq J_T\leq Ch_T^d.
$$

To justify the length comparisons, use the reference edge vectors to bound
$\|B_T\|$ by $Ch_T$, and bound $h_T$ by
$\|B_T\|\operatorname{diam}\widehat T$. A reference inscribed ball maps to an
ellipsoid containing a ball of radius $a_T\widehat\rho$, so
$\rho_T\geq a_T\widehat\rho$. Conversely, the width of $T$ in a left singular
direction for $a_T$ is at most $a_T\operatorname{diam}\widehat T$; an inscribed
ball has width $2\rho_T$ in that direction, so $\rho_T\leq Ca_T$.
The determinant bounds follow also from the volume of an inscribed ball and
a containing ball of radius $h_T$.

For $\widehat v=v\circ F_T$, change of variables and the chain rule give

$$
\begin{aligned}
\|v\|_{0,p,T}&=J_T^{1/p}\|\widehat v\|_{0,p,\widehat T},\\
|v|_{m,p,T}&\leq C J_T^{1/p}\|B_T^{-1}\|^m
|\widehat v|_{m,p,\widehat T},\\
|\widehat v|_{\ell,q,\widehat T}&\leq
C J_T^{-1/q}\|B_T\|^\ell|v|_{\ell,q,T}.
\end{aligned}
$$

The equality includes $p=\infty$, with $J_T^0=1$. The constants in the
derivative bounds account only for finite sums of components of derivative
tensors; they do not depend on the shape of $T$.

| Operation | Geometric factor |
|---|---|
| Measure a physical derivative of order $m$ | $\|B_T^{-1}\|^m\leq C\rho_T^{-m}$ |
| Pull back a derivative of order $\ell$ | $\|B_T\|^\ell\leq Ch_T^\ell$ |
| Change volume integrability from $q$ to $p$ | $J_T^{1/p-1/q}$ |

Shape regularity is the **additional** uniform bound $h_T/\rho_T\leq C$.
It implies $\rho_T\simeq h_T$ and $J_T\simeq h_T^d$. We will invoke it only
when passing from the explicit geometric estimates to their familiar
single-mesh-size forms.

### 2.2 Local seminorm estimates

```{prf:theorem} Local inverse inequality
:label: thm:lecture-07-inverse

Let $0\leq\ell\leq m$ be integers and $1\leq p,q\leq\infty$. For every
$v_h\in\mathbb P_k(T)$,

$$
\boxed{|v_h|_{m,p,T}\leq
C J_T^{1/p-1/q}h_T^\ell\rho_T^{-m}|v_h|_{\ell,q,T}.}
$$

The constant can depend on $k,m,\ell,p,q,d$ and the reference simplex, but
is independent of $T$, including its distortion. No ordering of $p$ and $q$
is required.

More generally, let $\widehat S$ be a fixed finite-dimensional subspace of
$W^{M,\infty}(\widehat T)$ for an integer $M\geq0$, and set
$S(T)=\{\widehat v\circ F_T^{-1}:\widehat v\in\widehat S\}$.
The same estimate holds for $v_h\in S(T)$ and integers
$0\leq\ell\leq m\leq M$, with a constant depending on $\widehat S$.
```

Notice that we are not using Sobolev-embedding restriction between these
indices: what we use is the fact that the functions belong to a fixed polynomial
space. The estimate remains true for $m>k$, when its left-hand side vanishes.
Its nontrivial derivative cases have $m\leq k$. If also $\ell>k$, both sides
vanish. These statements concern **cellwise** derivatives.

**Proof.** Write $\widehat v_h=v_h\circ F_T$. First prove on the fixed
reference simplex that

$$
|\widehat v_h|_{m,p,\widehat T}\leq
C|\widehat v_h|_{\ell,q,\widehat T}.
$$

For $\ell=0$, the right-hand side is a norm on $\mathbb P_k(\widehat T)$;
finite-dimensional norm equivalence controls every derivative on the left.
For $\ell\geq1$, its kernel is
$N_\ell=\mathbb P_k(\widehat T)\cap\mathbb P_{\ell-1}(\widehat T)$.
Since $m\geq\ell$, every element of $N_\ell$ also has zero order-$m$
seminorm. Choose any linear complement $Z$ of $N_\ell$ in
$\mathbb P_k(\widehat T)$. On $Z$ the right-hand side is a norm and controls
the left-hand seminorm by finite dimensionality. Subtracting the component
in $N_\ell$ leaves both seminorms unchanged, proving the reference estimate
for every polynomial. This accounts explicitly for the seminorm kernels.
For the general reference space, use the complement of
$\widehat S\cap\mathbb P_{\ell-1}(\widehat T)$ instead; the same argument
works because the reference simplex is connected. The assumed regularity
ensures that every seminorm used is finite, including the endpoint exponents.

Use the three affine transformation bounds from §2.1:

$$
\begin{aligned}
|v_h|_{m,p,T}
&\leq C J_T^{1/p}\|B_T^{-1}\|^m|\widehat v_h|_{m,p,\widehat T}\\
&\leq C J_T^{1/p}\|B_T^{-1}\|^m|\widehat v_h|_{\ell,q,\widehat T}\\
&\leq C J_T^{1/p-1/q}\|B_T^{-1}\|^m\|B_T\|^\ell
|v_h|_{\ell,q,T}\\
&\leq C J_T^{1/p-1/q}\rho_T^{-m}h_T^\ell|v_h|_{\ell,q,T}.
\end{aligned}
$$

The first line contributes physical derivatives and volume measure; the last
pullback contributes the lower derivative order and the inverse volume factor.
This proves the assertion, including the endpoint exponents. $\square$

Two cases used repeatedly are

$$
|v_h|_{1,p,T}\leq C\rho_T^{-1}\|v_h\|_{0,p,T},\qquad
\|v_h\|_{0,p,T}\leq CJ_T^{1/p-1/q}\|v_h\|_{0,q,T}.
$$

Only now impose shape regularity. Substituting $\rho_T\simeq h_T$ and
$J_T\simeq h_T^d$ yields the corollary

$$
|v_h|_{m,p,T}\leq
C h_T^{\ell-m+d(1/p-1/q)}|v_h|_{\ell,q,T}.
$$

A dilation by $\lambda$ multiplies $J_T$ by $\lambda^d$ and both $h_T$ and
$\rho_T$ by $\lambda$. Both sides therefore scale as $\lambda^{d/p-m}$.

### 2.3 Local full norm estimates

For integers $0\leq\ell\leq m$, $1\leq p,q\leq\infty$, and $h_T\leq1$,

$$
\boxed{\|v_h\|_{m,p,T}\leq
C J_T^{1/p-1/q}h_T^\ell\rho_T^{-m}\|v_h\|_{\ell,q,T}.}
$$

For each derivative order $j\geq\ell$, use §2.2 with $m=j$; since
$\rho_T\leq h_T\leq1$, the factor $h_T^\ell\rho_T^{-j}$ is at most
$h_T^\ell\rho_T^{-m}$. For $j<\ell$, the same-order estimate gives a factor
$h_T^j\rho_T^{-j}$. It too is bounded by $h_T^\ell\rho_T^{-m}$, because

$$
h_T^j\rho_T^{m-j}\leq h_T^m\leq h_T^\ell.
$$

Sum the finitely many derivative orders, or take their maximum. This proves
the full-norm bound without a shape assumption. Under shape regularity it
reduces to the usual factor $h_T^{\ell-m+d(1/p-1/q)}$.
The homogeneous seminorm estimate needs no restriction $h_T\leq1$.

One cannot reverse the derivative orders in the seminorm estimate without
additional constraints: for a nonconstant affine function $v_h$,
$|v_h|_{2,p,T}=0$ and $|v_h|_{1,p,T}>0$. The kernels differ.

### 2.4 Local estimates in Fractional norms

This extension is for independent study. For noninteger $s=r+\theta$,
$r\in\mathbb N_0$, $0<\theta<1$, and $1\leq p<\infty$, use the
Sobolev–Slobodeckij seminorm

$$
|v|_{s,p,D}^p=
\sum_{|\alpha|=r}\int_D\int_D
\frac{|D^\alpha v(x)-D^\alpha v(y)|^p}{|x-y|^{d+\theta p}}\,dx\,dy.
$$

The reference-space quotient argument still applies for real $0\leq t\leq s$
and finite $1\leq p,q<\infty$. For a positive order $a$, the polynomial kernel
is $\mathbb P_k\cap\mathbb P_{\lceil a\rceil-1}$, so the kernels are nested.
At order zero use the full norm, whose kernel is zero.

For noninteger $s=r+\theta$, changing both integration variables gives the
following explicit bounds:

$$
\begin{aligned}
|v|_{s,p,T}&\leq
C J_T^{2/p}\|B_T^{-1}\|^{s+d/p}|\widehat v|_{s,p,\widehat T},\\
|\widehat v|_{s,p,\widehat T}&\leq
C J_T^{-2/p}\|B_T\|^{s+d/p}|v|_{s,p,T}.
\end{aligned}
$$

Indeed, the double integral contributes $J_T^2$, the denominator is bounded
using $|B_T(\widehat x-\widehat y)|\geq
\|B_T^{-1}\|^{-1}|\widehat x-\widehat y|$, and the $r$ derivatives contribute
$\|B_T^{-1}\|^{rp}$. For the reverse bound use
$|\widehat x-\widehat y|\geq\|B_T\|^{-1}|x-y|$ and the pullback chain rule.

For example, if both $s,t$ are noninteger, the explicit inverse bound is

$$
|v_h|_{s,p,T}\leq
C J_T^{2/p-2/q}\rho_T^{-s-d/p}h_T^{t+d/q}|v_h|_{t,q,T}.
$$

If an order is integer, replace its transformation factor by the corresponding
integer factor from §2.1: the forward factor is $J_T^{1/p}\rho_T^{-s}$ and
the reverse factor is $J_T^{-1/q}h_T^t$. This also covers $t=0$.
These bounds retain cell distortion explicitly; they are not asserted to be
sharp anisotropic estimates.

Under shape regularity, both fractional and integer transformation factors
reduce respectively to $h_T^{d/p-s}$ and $h_T^{t-d/q}$. Consequently,

$$
|v_h|_{s,p,T}\leq
Ch_T^{t-s+d(1/p-1/q)}|v_h|_{t,q,T}.
$$

No noninteger $p=\infty$ convention is needed here; all endpoint statements
in §2.2 concern the ordinary integer-order Sobolev spaces. These inverse
inequalities are estimates for polynomials, and impose no regularity
assumption on the solution of the PDE.

### 2.5 Global estimates

For $v_h$ piecewise in $\mathbb P_k$, $1\leq p<\infty$, and integers
$0\leq\ell\leq m$, summation of the explicit local estimate gives

$$
\left(\sum_T|v_h|_{m,p,T}^p\right)^{1/p}
\leq C\left(\sum_T
h_T^{p\ell}\rho_T^{-pm}|v_h|_{\ell,p,T}^p\right)^{1/p}.
$$

The determinant factors cancel because the integrability exponents agree.
For $p=\infty$, replace the sums by maxima. These are **broken** seminorms:
a continuous piecewise polynomial is globally in $W^{1,p}$, but need not be
in $W^{m,p}$ for $m\geq2$. Higher distributional derivatives can contain
interface contributions.

For a conforming function, the gradient estimate gives directly

$$
|v_h|_{1,\Omega}\leq C
\left(\sum_T\rho_T^{-2}\|v_h\|_{0,T}^2\right)^{1/2}
\leq C\rho_{\min}^{-1}\|v_h\|_{0,\Omega},
\qquad \rho_{\min}=\min_T\rho_T.
$$

With shape regularity, $\rho_T\simeq h_T$, and the general weighted factor
becomes $h_T^{\ell-m}$. It can then be bounded by $h_{\min}^{\ell-m}$.
Replacing this by $Ch^{\ell-m}$ when $m>\ell$ additionally requires
quasi-uniformity. These are two distinct geometric assumptions.

Mixed-exponent local determinant factors cannot simply be replaced by a
single global mesh factor. For example, if $p<q$ and $m=\ell=0$, a global
factor $h^{d(1/p-1/q)}$ would tend to zero and fail on the constant function
$1$ on the fixed domain. Summation must also account for the exponents.

## 3. Polynomial traces follow from inverse inequalities

### 3.1 The role of Face Jacobians

Let $\widehat F$ be a face of $\widehat T$, $F=F_T(\widehat F)$, and
$\widehat n_F$ its reference unit normal. The constant surface Jacobian is

$$
\boxed{J_{T,F}=J_T\|B_T^{-T}\widehat n_F\|
=\frac{|F|}{|\widehat F|}.}
$$

This follows from the cofactor transformation of the oriented surface element;
taking its length gives the displayed scalar factor. Hence

$$
\|v\|_{0,p,F}=J_{T,F}^{1/p}
\|\widehat v\|_{0,p,\widehat F},\qquad 1\leq p\leq\infty.
$$

If $H_{T,F}$ is the altitude from the opposite vertex to $F$, then
$|T|=|F|H_{T,F}/d$ and the same identity holds on the reference simplex. Thus

$$
\frac{J_{T,F}}{J_T}
=\|B_T^{-T}\widehat n_F\|
=\frac{\widehat H_{\widehat F}}{H_{T,F}}
\leq C\rho_T^{-1}.
$$

The ratio is a reciprocal length, resolved separately for each face.
No shape assumption has been used.

### 3.2 Reference inverse estimates and polynomial trace bound

Choose the incenter $\widehat x_*$ of the fixed reference simplex and write
$\widehat b(\widehat x)=\widehat x-\widehat x_*$. Its normal component on
every reference face is the fixed inradius $\widehat\rho>0$.
For a smooth scalar function $\widehat v$ and $1<p<\infty$, the divergence
theorem and Hölder give

$$
\begin{aligned}
\widehat\rho\|\widehat v\|_{0,p,\partial\widehat T}^p
&=\int_{\widehat T}\widehat\nabla\cdot
\bigl(|\widehat v|^p\widehat b\bigr)\\
&\leq d\|\widehat v\|_{0,p,\widehat T}^p+
C_p\|\widehat v\|_{0,p,\widehat T}^{p-1}
|\widehat v|_{1,p,\widehat T}.
\end{aligned}
$$

Apply the reference inverse inequality to the polynomial $\widehat v_h$:
$|\widehat v_h|_{1,p,\widehat T}\leq C\|\widehat v_h\|_{0,p,\widehat T}$.
Since $\widehat\rho$ is fixed, this proves
$\|\widehat v_h\|_{0,p,\widehat F}\leq
C\|\widehat v_h\|_{0,p,\widehat T}$.
For $p=1$, regularize $|\widehat v|$ by
$(|\widehat v|^2+\varepsilon^2)^{1/2}$ in the divergence identity and pass to
the limit. For $p=\infty$, continuity of the polynomial bounds its boundary
supremum by its interior essential supremum.

Use also the order-zero reference inverse estimate to change from $p$ to $q$.
Transporting the resulting reference bound to $T$ gives

$$
\begin{aligned}
\|v_h\|_{0,p,F}
&=J_{T,F}^{1/p}\|\widehat v_h\|_{0,p,\widehat F}\\
&\leq C J_{T,F}^{1/p}\|\widehat v_h\|_{0,q,\widehat T}\\
&=C J_{T,F}^{1/p}J_T^{-1/q}\|v_h\|_{0,q,T}.
\end{aligned}
$$

```{prf:theorem} Discrete trace inequality with explicit geometry
:label: thm:lecture-07-polynomial-trace

For every $v_h\in\mathbb P_k(T)$, every face $F$, and $1\leq p,q\leq\infty$,

$$
\boxed{\|v_h\|_{0,p,F}\leq
C J_{T,F}^{1/p}J_T^{-1/q}\|v_h\|_{0,q,T}
\leq C\rho_T^{-1/p}J_T^{1/p-1/q}\|v_h\|_{0,q,T}.}
$$

The constant depends on the reference simplex, degree, and exponents, and is
independent of cell distortion. For $p<\infty$, the whole-boundary version is

$$
\|v_h\|_{0,p,\partial T}
\leq C\left(\sum_{F\subset\partial T}J_{T,F}\right)^{1/p}
J_T^{-1/q}\|v_h\|_{0,q,T}
\leq C\rho_T^{-1/p}J_T^{1/p-1/q}\|v_h\|_{0,q,T}.
$$

For $p=\infty$, take the maximum over faces and interpret all factors with
exponent $1/p$ as $1$.
```

In particular, with $p=q=2$,

$$
\|v_h\|_{0,F}\leq C(J_{T,F}/J_T)^{1/2}\|v_h\|_{0,T}
\leq C\rho_T^{-1/2}\|v_h\|_{0,T}.
$$

Applying the same trace bound to each polynomial derivative and the physical
inverse estimate yields

$$
\|\nabla v_h\|_{0,p,F}
\leq C(J_{T,F}/J_T)^{1/p}\rho_T^{-1}\|v_h\|_{0,p,T}.
$$

Under shape regularity, $J_{T,F}\simeq h_T^{d-1}$ and $J_T\simeq h_T^d$.
Only with this additional hypothesis does the mixed-exponent face factor
reduce to $h_T^{(d-1)/p-d/q}$, and for equal exponents to $h_T^{-1/p}$.

### 3.3 Autonomous scaled trace proof

For Scott–Zhang we also need traces of general Sobolev functions. The following
proof is independent of inverse inequalities and gives an alternative route
to the polynomial result.

```{prf:theorem} Cell trace inequality with explicit geometry
:label: thm:lecture-07-cell-trace

For every $1\leq p<\infty$, $v\in W^{1,p}(T)$, and face $F$,

$$
\boxed{\|v\|_{0,p,F}^p\leq C_p\frac{J_{T,F}}{J_T}
\bigl(\|v\|_{0,p,T}^p+h_T^p|v|_{1,p,T}^p\bigr).}
$$

In particular,

$$
\boxed{\|v\|_{0,p,\partial T}^p\leq C_p\rho_T^{-1}
\bigl(\|v\|_{0,p,T}^p+h_T^p|v|_{1,p,T}^p\bigr).}
$$

These constants are independent of cell distortion. With shape regularity,
$\rho_T\simeq h_T$, the last inequality becomes the usual scaled trace bound

$$
\|v\|_{0,p,\partial T}^p\leq C_p
\bigl(h_T^{-1}\|v\|_{0,p,T}^p+h_T^{p-1}|v|_{1,p,T}^p\bigr).
$$
```

**Autonomous proof on the reference simplex.** The divergence identity in §3.2
holds for smooth functions without any polynomial assumption. For $p>1$, use
Young's inequality with conjugate exponents $p/(p-1)$ and $p$ to obtain

$$
\|\widehat v\|_{0,p,\partial\widehat T}^p
\leq C_p\bigl(\|\widehat v\|_{0,p,\widehat T}^p+
|\widehat v|_{1,p,\widehat T}^p\bigr).
$$

For $p=1$, the regularized identity gives the sum of the two norms directly.
Smooth functions on the closure are dense in $W^{1,p}(\widehat T)$. Apply the
proved inequality to differences of smooth approximations: their boundary
restrictions are Cauchy in $\|\cdot\|_{0,p,\partial\widehat T}$. Their limit
defines the trace. The same bound proves independence of the approximation,
and passing to the limit establishes the reference estimate for every
$\widehat v\in W^{1,p}(\widehat T)$.

Now transport it to a single physical face. The exact volume and surface
changes of variables and the gradient pullback give

$$
\begin{aligned}
\|v\|_{0,p,F}^p
&=J_{T,F}\|\widehat v\|_{0,p,\widehat F}^p\\
&\leq C_p J_{T,F}
\bigl(\|\widehat v\|_{0,p,\widehat T}^p+
|\widehat v|_{1,p,\widehat T}^p\bigr)\\
&\leq C_p\frac{J_{T,F}}{J_T}
\bigl(\|v\|_{0,p,T}^p+\|B_T\|^p|v|_{1,p,T}^p\bigr)\\
&\leq C_p\frac{J_{T,F}}{J_T}
\bigl(\|v\|_{0,p,T}^p+h_T^p|v|_{1,p,T}^p\bigr).
\end{aligned}
$$

Summing over the fixed number of faces and using $J_{T,F}/J_T\leq C/\rho_T$
proves the whole-boundary bound. $\square$

For $p=\infty$, the Lipschitz representative on the convex simplex satisfies
$\|\gamma v\|_{0,\infty,\partial T}\leq\|v\|_{0,\infty,T}$. No infinite
power of a norm is used.

For $p=2$, the face estimate reads

$$
\|v\|_{0,F}^2\leq C\frac{J_{T,F}}{J_T}
\bigl(\|v\|_{0,T}^2+h_T^2|v|_{1,T}^2\bigr).
$$

The term $\|v\|_{0,T}$ is necessary, as a nonzero constant shows.
For the same reason a face-to-volume ratio is unavoidable: for $v=1$ the
squared norm ratio is exactly $|F|/|T|$.

### 3.4 Direct physical-cell proof

As a second autonomous proof of the whole-boundary estimate, choose the
incenter $x_*$ of $T$, so that $b(x)=x-x_*$ satisfies $b\cdot n=\rho_T$ on
every face and $|b|\leq h_T$. For smooth $v$ and $1<p<\infty$,

$$
\rho_T\|v\|_{0,p,\partial T}^p
\leq d\|v\|_{0,p,T}^p+
C_p h_T\|v\|_{0,p,T}^{p-1}|v|_{1,p,T}.
$$

Young's inequality applied to $\|v\|_{0,p,T}^{p-1}(h_T|v|_{1,p,T})$, followed
by division by $\rho_T$, gives the bound in §3.3. For $p=1$ use the regularized
identity; extend to general $W^{1,p}$ functions by density. This proof also
shows directly that $\rho_T|\partial T|=d|T|$ by taking the divergence of $b$.

Substituting the physical inverse gradient estimate into this continuous
trace bound would give the valid but coarser polynomial factor
$\rho_T^{-1/p}(1+(h_T/\rho_T)^p)^{1/p}$. The sharper polynomial bound in §3.2
comes from applying the inverse inequality **on the reference simplex before
transporting the trace**. It retains the face Jacobian and avoids this extra
distortion factor.

The abstract trace theorem also gives the reference estimate used in §3.3.
For $p=2$, its two continuity bounds are explicitly

$$
\|\gamma\widehat v\|_{0,\partial\widehat T}
\leq C\|\gamma\widehat v\|_{1/2,\partial\widehat T}
\leq C\|\widehat v\|_{1,\widehat T},
$$

and $\|\widehat v\|_{1,\widehat T}^2=
\|\widehat v\|_{0,\widehat T}^2+|\widehat v|_{1,\widehat T}^2$.

An interior face trace can be taken from either adjacent cell. For a global
$W^{1,p}$ function the two traces agree almost everywhere; a discontinuous
piecewise function has separate traces on the two sides.

**Geometry exercise.** Take the thin triangle with vertices $(0,0)$, $(1,0)$,
and $(0,\varepsilon)$, $0<\varepsilon\leq1$. For the standard reference
triangle, $B_T=\operatorname{diag}(1,\varepsilon)$ and $J_T=\varepsilon$.
Here $h_T\simeq1$ and $\rho_T\simeq\varepsilon$. On the horizontal unit edge
$J_{T,F}=1$, so $J_{T,F}/J_T=\varepsilon^{-1}$; on the vertical edge the ratio
is $1$. Verify that $v_h=y/\varepsilon$ has
$|v_h|_{1,T}/\|v_h\|_{0,T}\simeq\varepsilon^{-1}$. Thus $h_T$ alone cannot
control the inverse gradient or all face traces on distorted cells.

## 4. Global a priori estimates

From this point onward, assume a shape-regular family. The preceding explicit
geometry bounds then give constants uniform over all cells and permit the
use of $h_T$ alone. The patch arguments for Scott–Zhang use this assumption too.

Let $S_h\subset H^1(\Omega)$ be the continuous degree-$k$ Lagrange space and
$V_h=S_h\cap H_0^1(\Omega)$. First assume homogeneous data and exact assembly.
For Poisson, Galerkin orthogonality and Céa give

$$
a(u-u_h,v_h)=0\quad\forall v_h\in V_h,\qquad
|u-u_h|_1\leq\inf_{v_h\in V_h}|u-v_h|_1.
$$

For a general continuous coercive form, the last bound has factor $M/\alpha$.
If an admissible interpolant $J_hu\in V_h$ satisfies, for integer
$1\leq m\leq k+1$,

$$
|u-J_hu|_{1,T}\leq Ch_T^{m-1}|u|_{m,\omega_T},
$$

then a uniformly bounded overlap of the patches gives

```{math}
:label: eq:lecture-07-global-energy
|u-u_h|_{1,\Omega}\leq
C\left(\sum_T h_T^{2(m-1)}|u|_{m,\omega_T}^2\right)^{1/2}
\leq Ch^{m-1}|u|_{m,\Omega}.
```

For $m=1$ this gives boundedness, with no positive algebraic rate.
The construction below supplies the estimates for $k=1$, $m=1,2$, and the
fractional rates between them. The nodal construction of Lecture 3 required
additional regularity to make nodal evaluation bounded.

The Galerkin estimate in $\|\cdot\|_0$ uses an additional argument. If the homogeneous
adjoint has regularity $H^{1+t}$ and admits an energy approximation of order
$h^t$, [Lecture 6, §4](lecture-06.md)
gives

$$
\|u-u_h\|_0\leq Ch^t|u-u_h|_1.
$$

Thus primal regularity $H^{1+s}$ and adjoint shift $t$ yield $h^{s+t}$.
For linear triangles, $u\in H^2$ and a convex polygon give the familiar
energy rate $h$ and rate $h^2$ in $\|\cdot\|_0$. A smooth particular solution does not
supply regularity of the adjoint solution operator.

### 4.1 Boundary approximation produces a separate error term

For $g_h\in\gamma S_h$, let $u_h\in S_h$ have trace $g_h$ and satisfy
$a(u_h,v_h)=\langle f,v_h\rangle$ for all $v_h\in V_h$.
Choose any $G_h\in S_h$ with trace $g_h$. The trial set is $G_h+V_h$.

If $g_h=g$, the difference $u-u_h$ has zero trace. The proof of Céa applies
on this affine set and, for Poisson,

$$
|u-u_h|_1\leq\inf_{z_h\in G_h+V_h}|u-z_h|_1.
$$

For arbitrary $g\in H^{1/2}$, exact representability by $\gamma S_h$ generally
fails. A useful bound, valid also for $g_h\ne g$, is

```{math}
:label: eq:lecture-07-boundary-error
\boxed{\|u-u_h\|_1\leq C\left[
\inf_{z_h\in G_h+V_h}\|u-z_h\|_1
+\|g-g_h\|_{1/2,\partial\Omega}\right].}
```

**Proof.** For any $z_h$ in that set, $z_h-u_h\in V_h$. Orthogonality gives

$$
|z_h-u_h|_1^2=a(z_h-u,z_h-u_h)
\leq |z_h-u|_1|z_h-u_h|_1.
$$

Hence $|u-u_h|_1\leq2|u-z_h|_1$. For
$e=u-u_h$ and $\delta=g-g_h$, $e-E\delta\in H_0^1$. Poincaré and the lifting
bound imply

$$
\|e\|_1\leq C\bigl(|e|_1+\|\delta\|_{1/2,\partial\Omega}\bigr).
$$

Combine the estimates and take the infimum. $\square$

This derivation also explains why the energy seminorm alone does not control
the full error when its trace is nonzero. In the duality proof, the step of
testing with $e\in H_0^1$ is then unavailable. Boundary terms must be treated,
as in [Lecture 6, §5.3](lecture-06.md).
We do not infer $H^2$ regularity from $g\in H^{1/2}$.

## 5. Scott–Zhang replaces point values with local moments

We now take $k=1$. Write $S_h=\operatorname{span}\{\phi_i\}_{i=1}^N$, where
$\phi_i$ is the global hat function associated with vertex $a_i$.
An expression $\sum_i v(a_i)\phi_i$ is unavailable for general $v\in H^1$.
An edge trace, however, belongs to $L^2$ and can be integrated.

The original Scott–Zhang construction uses dual polynomial moments on selected
cells or faces. It is a projection and preserves representable boundary data;
see {cite:ts}`ScottZhang1990`. We give an explicit two-dimensional linear
construction and prove the estimates needed here.

### 5.1 A dual basis on a selected edge

For every vertex $a_i$, choose one mesh edge $F_i$ containing it. For a boundary
vertex, choose a **boundary edge**. Fix these choices for the operator.
On $F_i=[a_i,a_j]$, set $L_i=|F_i|$ and use the coordinate $t\in[0,1]$ with
$t=0$ at $a_i$. The restricted nodal functions are
$\lambda_i=1-t$ and $\lambda_j=t$. Their mass matrix and inverse are

$$
M_{F_i}=\frac{L_i}{6}\begin{pmatrix}2&1\\1&2\end{pmatrix},\qquad
M_{F_i}^{-1}=\frac1{L_i}\begin{pmatrix}4&-2\\-2&4\end{pmatrix}.
$$

The dual polynomial selecting the coefficient at $a_i$ is

$$
\psi_i=\frac{4\lambda_i-2\lambda_j}{L_i}=\frac{4-6t}{L_i},\qquad
\int_{F_i}\psi_i\lambda_i=1,\quad
\int_{F_i}\psi_i\lambda_j=0.
$$

In particular, $\|\psi_i\|_{0,F_i}=2L_i^{-1/2}$.
Define

```{math}
:label: eq:lecture-07-sz-definition
\ell_i(v)=\int_{F_i}\psi_i\,\gamma_{F_i}v,\qquad
\boxed{J_hv=\sum_{i=1}^N\ell_i(v)\phi_i.}
```

For an interior edge, either adjacent cell defines the same trace of a global
$H^1$ function. Thus $J_h:H^1(\Omega)\to S_h$ is well defined and linear.
This face-based operator is not defined on arbitrary $L^2(\Omega)$ classes:
they have no edge trace.

### 5.2 Projection and boundary preservation follow from duality

**Projection.** For $v_h=\sum_j c_j\phi_j$, its trace on $F_i$ is
$c_i\lambda_i+c_j\lambda_j$. Therefore $\ell_i(v_h)=c_i$, so

$$
J_hv_h=v_h\quad\forall v_h\in S_h,\qquad J_h^2=J_h.
$$

In particular, every global affine polynomial is reproduced.

**Homogeneous boundary data.** If $\gamma v=0$, then $\ell_i(v)=0$ at every
boundary vertex, because its chosen edge lies in $\partial\Omega$. The
piecewise linear trace of $J_hv$ has zero values at all boundary vertices;
it therefore vanishes on every boundary edge. Hence

$$
J_h(H_0^1(\Omega))\subset V_h.
$$

**Representable nonhomogeneous data.** If $\gamma v=g_h\in\gamma S_h$, choose
$w_h\in S_h$ with that trace. At boundary vertices the moments of $v$ and
$w_h$ coincide. Consequently $\gamma J_hv=g_h$.
For general $g$, the boundary trace is a moment approximation $Q_hg$, not an
exact copy of $g$. Its definition uses only $g$, independently of the interior
extension, and $Q_h$ is a projection onto $\gamma S_h$.

### 5.3 Locality is measured by vertex patches

Let $\omega_T$ be the interior, relative to $\Omega$, of the union of the
closures of all triangles sharing at least one vertex with $T$. Integrals and
Sobolev norms use its open interior. Only the moments associated with vertices
of $T$ affect $J_hv|_T$. Every selected edge for these vertices belongs to this
patch. Thus changing $v$ outside $\omega_T$ leaves $J_hv|_T$ unchanged.

Shape regularity in two dimensions bounds the number of triangles at a vertex
by the minimum-angle bound. Consecutive triangles in a vertex star share an
edge and have comparable diameters; the bounded chain length makes all their
sizes comparable. Consequently,

- $\operatorname{diam}\omega_T\leq Ch_T$;
- the number of triangles in $\omega_T$ and the overlap multiplicity of all
  patches are uniformly bounded;
- cellwise scaling and Poincaré estimates on these patches have uniform constants.

These are local properties of conforming triangular meshes. They allow strong
global grading.

## 6. Stability and approximation follow from moment bounds

### 6.1 A preliminary local bound retains a zeroth-order term

For a vertex $a_i$ of $T$, take a triangle $K_i\subset\omega_T$ adjacent to
$F_i$. The dual norm, Cauchy–Schwarz, and the scaled trace estimate give

$$
\begin{aligned}
|\ell_i(v)|
&\leq 2L_i^{-1/2}\|v\|_{0,F_i}\\
&\leq C\bigl(h_T^{-1}\|v\|_{0,K_i}+|v|_{1,K_i}\bigr).
\end{aligned}
$$

On $T$, $\|\phi_i\|_{0,T}\leq Ch_T$ and
$|\phi_i|_{1,T}\leq C$. Summing its three coefficients yields

```{math}
:label: eq:lecture-07-preliminary-stability
\begin{aligned}
\|J_hv\|_{0,T}&\leq
C\bigl(\|v\|_{0,\omega_T}+h_T|v|_{1,\omega_T}\bigr),\\
|J_hv|_{1,T}&\leq
C\bigl(h_T^{-1}\|v\|_{0,\omega_T}+|v|_{1,\omega_T}\bigr).
\end{aligned}
```

The $h_T^{-1}$ term must be removed to obtain uniform energy stability.

### 6.2 Polynomial subtraction removes the unwanted term

We use the following patch estimates, with $c$ a suitable constant and $q$
a suitable affine polynomial:

```{math}
:label: eq:lecture-07-patch-approximation
\begin{aligned}
\|v-c\|_{0,\omega_T}&\leq Ch_T|v|_{1,\omega_T},\\
\|v-q\|_{0,\omega_T}+h_T|v-q|_{1,\omega_T}
&\leq Ch_T^2|v|_{2,\omega_T}.
\end{aligned}
```

**Justification of uniformity.** On each triangle use the scaled Poincaré
inequality with its cell mean. For neighboring cells $K,L$, continuity of the
trace on their common edge $F$ gives

$$
|v_K-v_L|\leq |F|^{-1/2}
\bigl(\|v-v_K\|_{0,F}+\|v-v_L\|_{0,F}\bigr)
\leq C\bigl(|v|_{1,K}+|v|_{1,L}\bigr).
$$

The last step uses the cell trace and cell Poincaré inequalities and local
size comparability. Connect the cells in $\omega_T$ by a uniformly bounded
number of edge neighbors and compare their means to one fixed cell mean.
Multiplication by each cell's area square root supplies $h_T$, proving the
first inequality. For $v\in H^2$, apply it to each component of $\nabla v$
to select a constant vector $b$ with
$\|\nabla v-b\|_{0,\omega_T}\leq Ch_T|v|_{2,\omega_T}$.
Apply it again to $v-b\cdot x$ and select its constant part. This gives the
second inequality with $q=b\cdot x+c$. This argument proves the required
patch estimates using the mesh geometry, without assuming a single affine
reference map for a whole patch.

Since $J_hc=c$,

$$
|J_hv|_{1,T}=|J_h(v-c)|_{1,T}
\leq C|v|_{1,\omega_T}.
$$

This proves the local seminorm stability. The preliminary bound in $\|\cdot\|_0$ and
bounded patch overlap also give

$$
\boxed{\|J_hv\|_{1,\Omega}\leq C\|v\|_{1,\Omega}.}
$$

The proof does not give a bound $\|J_hv\|_0\leq C\|v\|_0$ on arbitrary
$H^1$ inputs; the selected edge moments require trace regularity.

```{prf:theorem} Local Scott–Zhang estimates for linear triangles
:label: thm:lecture-07-sz-local

For $v\in H^1(\omega_T)$,

$$
\|v-J_hv\|_{0,T}\leq Ch_T|v|_{1,\omega_T},\qquad
|v-J_hv|_{1,T}\leq C|v|_{1,\omega_T}.
$$

For $v\in H^2(\omega_T)$,

$$
\|v-J_hv\|_{0,T}+h_T|v-J_hv|_{1,T}
\leq Ch_T^2|v|_{2,\omega_T}.
$$
```

**Proof.** Subtract the constant $c$ or the affine polynomial $q$ from
{eq}`eq:lecture-07-patch-approximation`. Polynomial reproduction gives
$v-J_hv=(v-q)-J_h(v-q)$, and similarly for $c$. The triangle inequality and
{eq}`eq:lecture-07-preliminary-stability` give both assertions. $\square$

### 6.3 Edge error estimates follow from the cell trace inequality

For a mesh edge $F$, choose an adjacent triangle $T$ and put
$\omega_F=\omega_T$. Shape regularity gives $h_F\simeq h_T$.
Apply the cell trace inequality to $v-J_hv\in H^1(T)$ and the preceding
local estimates. For $m=1,2$,

```{math}
:label: eq:lecture-07-sz-edge
\boxed{\|v-J_hv\|_{0,F}\leq
Ch_F^{m-1/2}|v|_{m,\omega_F}.}
```

In particular, for $v\in H^1$,

$$
\|v-J_hv\|_{0,T}\leq Ch_T|v|_{1,\omega_T},\qquad
\|v-J_hv\|_{0,F}\leq Ch_F^{1/2}|v|_{1,\omega_F}.
$$

These two weights will be used to pair cell and face residuals with an error
of interpolation in the a posteriori analysis.

## 7. Scott–Zhang completes the low-regularity a priori argument

For homogeneous data, $J_hu\in V_h$ for every $u\in H_0^1$. For linear
triangles and $u\in H^2\cap H_0^1$, Céa and the local estimate immediately give

$$
|u-u_h|_1\leq C
\left(\sum_T h_T^2|u|_{2,\omega_T}^2\right)^{1/2}
\leq Ch|u|_{2,\Omega}.
$$

### 7.1 Fractional regularity follows by interpolation of operators

This subsection is for independent study. It supplies the global fractional
approximation assumption used in Lecture 6; no pointwise interpolation is
introduced.

Let $R_h=I-J_h$. Summing the local estimates gives endpoint bounds

$$
\begin{array}{ll}
\|R_hv\|_1\leq C\|v\|_1,&
\|R_hv\|_1\leq Ch\|v\|_2,\\
\|R_hv\|_0\leq Ch\|v\|_1,&
\|R_hv\|_0\leq Ch^2\|v\|_2.
\end{array}
$$

Use the real interpolation identity
$(H^1(\Omega),H^2(\Omega))_{s,2}=H^{1+s}(\Omega)$, $0<s<1$, with equivalent
norms on this fixed Lipschitz domain, and boundedness of linear operators on
interpolation spaces. These are Sobolev interpolation results; see
{cite:ts}`BrennerScott2010`. Interpolating the two bounds with the same target
space gives

$$
\boxed{\|v-J_hv\|_1\leq Ch^s\|v\|_{1+s},\qquad
\|v-J_hv\|_0\leq Ch^{1+s}\|v\|_{1+s}.}
$$

The first operator norm is bounded by $C\cdot1^{1-s}h^s$ and the second by
$C\cdot h^{1-s}(h^2)^s$. The identity above can be obtained using a common
bounded extension for the integer endpoint spaces and interpolation on
$\mathbb R^2$. Constants may depend on $s$ and $\Omega$.

Apply this result on the **full** spaces and then take a function with zero
trace. Boundary preservation makes its interpolant admissible. This avoids
assuming an interpolation identity for spaces with boundary constraints.
Céa now proves

$$
|u-u_h|_1\leq Ch^s\|u\|_{1+s},\qquad 0<s\leq1.
$$

If the homogeneous adjoint shift is $t$, the same construction supplies its
approximation of order $h^t$, hence $\|u-u_h\|_0\leq
Ch^{s+t}\|u\|_{1+s}$. The corner thresholds from Lecture 6 remain strict.

For a solution only in $H_0^1$, the operator still gives convergence without
an asserted algebraic rate. Indeed, choose $w\in C_c^\infty(\Omega)$ close to
$u$ in $H^1$. Uniform stability and the $H^2$ estimate give

$$
\|u-J_hu\|_1\leq C\|u-w\|_1+Ch\|w\|_2.
$$

First make the first term small by density, then send $h$ to zero.
Céa gives convergence of the Galerkin solution in the energy norm.

### 7.2 The boundary operator gives a compatible discrete lifting

Define $Q_hg=\gamma J_hEg$. Boundary moments show that $Q_hg$ is independent
of the lifting $Eg$. Trace continuity and global stability yield

$$
\|Q_hg\|_{1/2,\partial\Omega}\leq C\|g\|_{1/2,\partial\Omega}.
$$

If $g_h\in\gamma S_h$, the function $J_hEg_h\in S_h$ has trace $g_h$ and
norm $\|\cdot\|_1$ bounded by $C\|g_h\|_{1/2}$. It is thus a uniformly bounded
discrete lifting. This is a consequence of the boundary-preserving construction.

For a continuous solution with trace $g$, choose $g_h=Q_hg$. Then $J_hu$ has
trace $g_h$, and trace continuity gives

$$
\|g-g_h\|_{1/2,\partial\Omega}
\leq C\|u-J_hu\|_{1,\Omega}.
$$

Insert $z_h=J_hu$ in {eq}`eq:lecture-07-boundary-error`. For
$u\in H^{1+s}(\Omega)$, $0<s\leq1$, this proves

$$
\|u-u_h\|_1\leq Ch^s\|u\|_{1+s}.
$$

This extension specifies **which boundary approximation** is used. Other
choices of $g_h$, including nodal interpolation of smooth data, require their
own approximation estimates.

### 7.3 Higher fixed polynomial degrees

For degree $k>1$, replace the two edge polynomials by the local Lagrange basis
on a selected edge, and use the inverse of its mass matrix to define the dual
moments. Nodes in the interior of a triangle require cell moments on that
triangle. Nodes on the boundary must use boundary edges. The same moment
argument proves projection and preservation of representable boundary data.

For integer $1\leq m\leq k+1$, the resulting operator has local approximation
bounds of orders $h_T^m$ in $\|\cdot\|_{0,T}$ and $h_T^{m-1}$ in $|\cdot|_{1,T}$, with constants
allowed to depend on $k$. Full higher-degree estimates are an external result;
see {cite:ts}`ScottZhang1990`. The complete proof in this lecture covers $P_1$.

## 8. Exercises

1. **Lifting independence.** Given two extensions $G_1,G_2$ of the same trace,
   express the two homogeneous unknowns $w_1,w_2$ and verify
   $w_2=w_1+G_1-G_2$ directly in the weak equation.
2. **Dual edge moments.** Integrate $(4-6t)(1-t)$ and $(4-6t)t$ over $[0,1]$.
   Recover the mass matrix inverse and the norm $2|F|^{-1/2}$ of the first dual
   polynomial. Explain why averaging with $|F|^{-1}$ alone fails to reproduce
   the coefficient at a specified endpoint.
3. **Boundary choices.** For a boundary vertex, replace its chosen boundary edge
   by an interior edge. Construct a smooth function with zero boundary trace
   and a nonzero moment on that edge. Identify the failure of boundary
   preservation.
4. **Global mesh factors.** On a graded family, compare the weighted inverse
   bound with the bound involving $h_{\min}$. Explain why the global a priori
   upper estimate uses $h_{\max}$ without requiring quasi-uniformity.
5. **A priori chain.** Assume a primal solution in $H^{1+s}$ and an adjoint shift
   $t$, both with zero trace. Identify the separate uses of Scott–Zhang, Céa,
   and duality in obtaining the energy and $\|\cdot\|_0$ rates. State what changes if
   the imposed trace is $g_h\ne g$.

## Further reading

The trace, scaling, and approximation background is treated in
{cite:ts}`BrennerScott2010` and {cite:ts}`ErnGuermond2004`.
The general integer-order inverse and discrete trace estimates are also treated
in [Guermond, Chapter 12, §§12.1–12.2](https://people.tamu.edu/~guermond/M661_FALL_2021/chap12.pdf).
The original interpolation construction is L. Ridgway Scott and Shangyou Zhang,
[Finite element interpolation of nonsmooth functions satisfying boundary conditions](https://doi.org/10.1090/S0025-5718-1990-1011446-7),
*Mathematics of Computation* **54** (1990), 483–493.
