# Lecture 6 — Elliptic regularity and error estimates in weaker norms

## Objectives and prerequisites

We connect three steps: the boundary-value problem determines solution
regularity; regularity determines approximation, i.e., a-priori error estimates
in the $H^1$ norm; approximation of the adjoint solution controls Galerkin error
in a weaker norm. By the end, you should be able to derive a corner singularity,
interpret fractional Sobolev regularity, and prove the Aubin–Nitsche estimate
for Poisson.

We use the weak formulation and Galerkin orthogonality from
[Lecture 1](lecture-01.md), Sobolev norms and mesh geometry from
[Lecture 2](lecture-02.md), the approximation estimates from
[Lecture 3](lecture-03.md), and conforming Lagrange spaces from
[Lecture 4](lecture-04.md). [Lecture 5](lecture-05.md) supplies the assembly
operations used in the reproducible experiment below.

The main analysis concerns bounded polygonal domains in two dimensions,
full Dirichlet conditions, fixed polynomial degree, exact assembly and
shape-regular conforming affine meshes. We state additional assumptions when
needed. Constants are independent of the mesh size; they may depend on the
fixed domain, degree, shape-regularity bound and the stated regularity constants.

## 1. Regularity

Let $\Omega\subset\mathbb R^2$ be bounded, connected and Lipschitz. The weak
homogeneous Dirichlet problem is

$$
u\in H_0^1(\Omega),\qquad
 a(u,v)=\langle f,v\rangle\quad\forall v\in H_0^1(\Omega),
\qquad a(w,v)=\int_\Omega\nabla w\cdot\nabla v.
$$

For $f\in H^{-1}(\Omega):=(H_0^1(\Omega))'$, Lax–Milgram gives a unique weak
solution and $|u|_{H^1}\leq C\|f\|_{H^{-1}}$. A separate elliptic regularity
theorem is needed to conclude that $u$ has two derivatives in $L^2$.

For **Poisson with full homogeneous Dirichlet conditions on a bounded convex
polygon**, the regularity theorem gives

$$
u\in H^2(\Omega)\cap H_0^1(\Omega),\qquad
\|u\|_{H^2(\Omega)}\leq C_{\rm reg}\|f\|_{L^2(\Omega)}.
$$

We use this PDE theorem as an external result; see
{cite:ts}`BrennerScott2010` and {cite:ts}`ErnGuermond2004`. Convexity here does
not imply arbitrary higher regularity. Other operators, discontinuous
coefficients, mixed boundary conditions or different domains require their own
regularity statements. Even a smooth domain can lose regularity at a junction
of Dirichlet and Neumann boundary conditions.

## 2. A re-entrant corner produces a singular gradient

For $\pi<\omega<2\pi$, set $\theta_0=2\pi-\omega$ and consider

$$
\Omega_\omega=\{(r\cos\theta,r\sin\theta):0<r<1,
                      \theta_0<\theta<2\pi\}.
$$

```{figure} ../assets/generated/lecture-06/pacman-geometry.svg
:alt: A sector with interior angle 3 pi over 2, its re-entrant corner, radial coordinate, and boundary rays.
:width: 55%

The interior angle is $\omega$. The removed sector has angle $2\pi-\omega$.
The numerical experiment below replaces the outer arc by a fixed polygon.
```

### 2.1 Differential operators in polar coordinates

The coordinate map and its orthonormal basis are

$$
F(r,\theta)=(r\cos\theta,r\sin\theta),\qquad
\boldsymbol e_r=(\cos\theta,\sin\theta),\qquad
\boldsymbol e_\theta=(-\sin\theta,\cos\theta).
$$

All formulas below hold for $r>0$, on a fixed angular branch, for $C^2$ scalar
functions and $C^1$ vector fields. We write
$u(r,\theta)$ for the Cartesian function composed with $F$. Since

$$
\partial_rF=\boldsymbol e_r,\qquad
\partial_\theta F=r\boldsymbol e_\theta,
$$

the chain rule gives $u_r=\nabla u\cdot\boldsymbol e_r$ and
$u_\theta=r\nabla u\cdot\boldsymbol e_\theta$. To reconstruct the gradient,
write $\nabla u=a\boldsymbol e_r+b\boldsymbol e_\theta$. The identities
$\boldsymbol e_r\cdot\boldsymbol e_r=\boldsymbol e_\theta\cdot\boldsymbol e_\theta=1$
and $\boldsymbol e_r\cdot\boldsymbol e_\theta=0$ imply

$$
\begin{aligned}
\nabla u\cdot\boldsymbol e_r
&=a(\boldsymbol e_r\cdot\boldsymbol e_r)
 +b(\boldsymbol e_\theta\cdot\boldsymbol e_r)=a=u_r,\\
\nabla u\cdot\boldsymbol e_\theta
&=a(\boldsymbol e_r\cdot\boldsymbol e_\theta)
 +b(\boldsymbol e_\theta\cdot\boldsymbol e_\theta)=b=\frac{u_\theta}{r}.
\end{aligned}
$$

Substituting these two coefficients into the expansion gives

$$
\boxed{\nabla u=u_r\boldsymbol e_r+\frac{u_\theta}{r}\boldsymbol e_\theta.}
$$

In Cartesian components, this is equivalent to the differential operators

$$
\partial_x=\cos\theta\,\partial_r-\frac{\sin\theta}{r}\,\partial_\theta,
\qquad
\partial_y=\sin\theta\,\partial_r+\frac{\cos\theta}{r}\,\partial_\theta.
$$

For a vector field $\boldsymbol q=q_r\boldsymbol e_r+q_\theta\boldsymbol e_\theta$,
the coefficients $q_r,q_\theta$ are its components in the **unit** polar basis.
This basis depends on $\theta$:

$$
\partial_\theta\boldsymbol e_r=\boldsymbol e_\theta,\qquad
\partial_\theta\boldsymbol e_\theta=-\boldsymbol e_r,
\qquad \partial_r\boldsymbol e_r=\partial_r\boldsymbol e_\theta=0.
$$

Consequently,

$$
\begin{aligned}
\partial_r\boldsymbol q
&=(\partial_rq_r)\boldsymbol e_r+(\partial_rq_\theta)\boldsymbol e_\theta,\\
\partial_\theta\boldsymbol q
&=(\partial_\theta q_r-q_\theta)\boldsymbol e_r
 +(q_r+\partial_\theta q_\theta)\boldsymbol e_\theta.
\end{aligned}
$$

Using the Cartesian differential operators above,

$$
\begin{aligned}
\nabla\cdot\boldsymbol q
&=\boldsymbol e_r\cdot\partial_r\boldsymbol q
  +\frac1r\boldsymbol e_\theta\cdot\partial_\theta\boldsymbol q\\
&=\partial_rq_r+\frac{q_r}{r}+\frac1r\partial_\theta q_\theta\\
&=\boxed{\frac1r\partial_r(rq_r)+\frac1r\partial_\theta q_\theta}.
\end{aligned}
$$

The term $q_r/r$ comes from differentiating the angular basis vectors.
For a twice differentiable scalar function, take
$\boldsymbol q=\nabla u$, so $q_r=u_r$ and $q_\theta=u_\theta/r$. This gives

$$
\Delta u=\nabla\cdot\nabla u
=\frac1r\partial_r(ru_r)+\frac1r\partial_\theta\left(\frac{u_\theta}{r}\right)
=u_{rr}+\frac1r u_r+\frac1{r^2}u_{\theta\theta}.
$$

### 2.2 The angular boundary conditions select the exponent

Seek a separated harmonic function $u=r^\lambda\Phi(\theta)$. Substitution gives

$$
\Delta(r^\lambda\Phi)=r^{\lambda-2}(\lambda^2\Phi+\Phi''),
\qquad \Phi(\theta_0)=\Phi(2\pi)=0.
$$

Choose $\Phi(\theta)=\sin\bigl(\alpha(\theta-\theta_0)\bigr)$ with $\alpha>0$.
Then $\Phi''=-\alpha^2\Phi$, so

$$
\Delta\bigl(r^\lambda\Phi(\theta)\bigr)
=(\lambda^2-\alpha^2)r^{\lambda-2}\Phi(\theta).
$$

Taking the positive radial exponent $\lambda=\alpha$ therefore makes the
function harmonic. The condition $\Phi(\theta_0)=0$ holds automatically, while
the other ray requires

$$
\Phi(2\pi)=\sin\bigl(\alpha(2\pi-\theta_0)\bigr)
=\sin(\alpha\omega)=0,
\qquad \alpha=\frac{n\pi}{\omega},\quad n=1,2,\ldots.
$$

The smallest positive choice is $n=1$, hence $\lambda=\alpha=\pi/\omega$. Thus

```{math}
:label: eq:lecture-06-corner
u(r,\theta)=r^\alpha\sin\bigl(\alpha(\theta-\theta_0)\bigr),
\qquad \alpha=\frac\pi\omega\in(1/2,1).
```

This solves $-\Delta u=0$, with zero data on the rays and
$g(\theta)=\sin(\alpha(\theta-\theta_0))$ on $r=1$. The nonzero outer trace is
essential: a harmonic $H_0^1$ solution would be zero by uniqueness.

### 2.3 The first two Sobolev orders

The radial and angular derivatives and the gradient formula above give

$$
|\nabla u|^2=\alpha^2r^{2\alpha-2},\qquad
\int_{\Omega_\omega}|\nabla u|^2
=\omega\alpha^2\int_0^1r^{2\alpha-1}\,dr=\frac{\omega\alpha}{2}.
$$

Hence $u\in H^1(\Omega_\omega)$. To check the next order, we compute the
Cartesian Hessian explicitly, retaining the constant phase $-\alpha\theta_0$.
The Cartesian differential operators from Section 2.1 give

$$
u_x=\alpha r^{\alpha-1}\sin\bigl((\alpha-1)\theta-\alpha\theta_0\bigr),
\qquad
u_y=\alpha r^{\alpha-1}\cos\bigl((\alpha-1)\theta-\alpha\theta_0\bigr).
$$

Differentiating once more and setting
$\psi=(\alpha-2)\theta-\alpha\theta_0$, we obtain

$$
D^2u=\alpha(\alpha-1)r^{\alpha-2}
\begin{pmatrix}
\sin\psi & \cos\psi\\
\cos\psi & -\sin\psi
\end{pmatrix}.
$$

In the squared Frobenius norm, the angular factor is
$2\sin^2\psi+2\cos^2\psi=2$. Thus

$$
|D^2u|_F^2=2\alpha^2(1-\alpha)^2r^{2\alpha-4},
\qquad |D^2u|_F^2=u_{xx}^2+2u_{xy}^2+u_{yy}^2.
$$

The radial integral $\int_0^1r^{2\alpha-3}\,dr$ diverges for $\alpha<1$.
Consequently $u\notin H^2$. The failure occurs at the corner despite smooth
coefficients and a zero forcing term.

Multiplying this function by a smooth radial cutoff that equals one near the
corner and vanishes near $r=1$ gives a homogeneous Dirichlet example. Its forcing
$-\Delta(\chi u)$ is supported away from the corner and belongs to $L^2$;
the local singularity is unchanged. This shows directly why the convex-domain
$H^2$ theorem cannot be transferred to a re-entrant domain.

For $\omega=\pi$, $\alpha=1$ and the displayed function is $u=-y$.
The Hessian coefficient vanishes: this is an affine solution, with no corner
singularity. It will provide a useful numerical control.

## 3. Fractional Sobolev spaces quantify the missing derivative

### 3.1 Fourier weights on the whole space

Use the unitary Fourier transform

$$
\widehat v(\xi)=(2\pi)^{-d/2}\int_{\mathbb R^d}e^{-ix\cdot\xi}v(x)\,dx.
$$

For $s\in\mathbb R$, define

$$
H^s(\mathbb R^d)=\{v\in\mathcal S'(\mathbb R^d):
 (1+|\xi|^2)^{s/2}\widehat v\in L^2(\mathbb R^d)\},\qquad
\|v\|_{H^s}^2=\int_{\mathbb R^d}(1+|\xi|^2)^s|\widehat v|^2.
$$

For nonnegative integer $s$, Plancherel and
$\widehat{D^\beta v}=(i\xi)^\beta\widehat v$ identify this norm, up to constants,
with the usual derivative norm. Noninteger weights measure intermediate decay
of the Fourier transform. The spaces with negative $s$ are defined by the same
weighted formula on the whole space. On a bounded domain, negative-order spaces
require a specified dual pairing and boundary convention.

### 3.2 Difference quotients on a domain

For $0<s<1$ and an open set $D\subset\mathbb R^d$, define

$$
|v|_{H^s(D)}^2=\int_D\int_D\frac{|v(x)-v(y)|^2}{|x-y|^{d+2s}}\,dx\,dy,
\qquad \|v\|_{H^s(D)}^2=\|v\|_{L^2(D)}^2+|v|_{H^s(D)}^2.
$$

For an integer $m\geq0$, the derivative definition of $H^{m+s}(D)$ requires
$v\in H^m(D)$ and $D^\beta v\in H^s(D)$ for every $|\beta|=m$.

On a bounded Lipschitz domain these spaces agree, with equivalent norms, with
restrictions of the whole-space Fourier spaces. The restriction norm is the
infimum of $\|Ev\|_{H^{m+s}(\mathbb R^d)}$ over all extensions $Ev$.
The equivalence uses a bounded Sobolev extension operator. Extension by zero
requires additional boundary hypotheses and is not the extension used here.
For $0<s<1$, details and proofs are given by
[Di Nezza, Palatucci and Valdinoci](https://arxiv.org/abs/1104.4345), Sections 2,
3 and 5. Higher orders follow by the Sobolev extension theorem and derivatives;
see {cite:ts}`BrennerScott2010`.

### 3.3 Real interpolation fills the interval between two spaces

For a continuously embedded pair $X_1\hookrightarrow X_0$, use the quadratic
$K$ functional

$$
K_2(t,v)^2=\inf_{v=v_0+v_1}
 \bigl(\|v_0\|_{X_0}^2+t^2\|v_1\|_{X_1}^2\bigr),\qquad t>0.
$$

It is equivalent to the usual $K$ functional with a sum of norms, since
$(a^2+b^2)^{1/2}\leq a+b\leq\sqrt2(a^2+b^2)^{1/2}$. The interpolation space
$(X_0,X_1)_{s,2}$ consists of those $v\in X_0$ for which

$$
\int_0^\infty t^{-2s}K_2(t,v)^2\,\frac{dt}{t}<\infty.
$$

For the unrestricted Sobolev spaces on a bounded Lipschitz domain,

$$
(L^2(D),H^1(D))_{s,2}=H^s(D),\qquad 0<s<1,
$$

with equivalent norms. Replacing $H^1(D)$ by $H_0^1(D)$ changes the interpolation
space through boundary conditions, including a special critical space at
$s=1/2$. We will not make that replacement implicitly.

The three constructions provide different tools: Fourier weights describe
frequency decay, the double integral detects spatial differences, and real
interpolation transfers operator bounds between regularity levels.

### 3.4 Corner regularity has a strict threshold

For the noninteger corner exponent in {eq}`eq:lecture-06-corner`,

$$
u\in H^{1+s}(\Omega_\omega)\quad\text{for every }0<s<\alpha,
\qquad u\notin H^{1+\alpha}(\Omega_\omega).
$$

Here is the scaling behind this threshold. Write $g=\nabla u$, homogeneous of
degree $\alpha-1$. On dyadic annuli
$A_j=\Omega_\omega\cap\{2^{-j-1}<r<2^{-j}\}$, the fractional seminorm rescales as

$$
|g|_{H^s(A_j)}^2
=2^{-2j(\alpha-s)}|g|_{H^s(A_0)}^2.
$$

The factor follows from two area measures, the denominator of degree $2+2s$,
and the squared function of degree $2\alpha-2$. At $s=\alpha$ each annulus
contributes the same positive amount, proving divergence. Positivity follows
because $g$ is nonconstant on $A_0$.

For completeness, interactions between annuli also give a finite total for
$s<\alpha$. Choose a fixed $c_\omega>0$ small enough that pairs with
$|x-y|<c_\omega|x|$ do not cross the angular branch cut. On such pairs,
$|g(x)-g(y)|\leq C_\omega |x|^{\alpha-2}|x-y|$; integrating the difference
quotient in $y$ bounds its contribution by
$C|x|^{2\alpha-2-2s}$. On the remaining pairs, use
$|g(x)-g(y)|^2\leq2|g(x)|^2+2|g(y)|^2$ and exchange $x,y$ for the second term.
The radii are bounded by a constant times $|x-y|$ there, so the same bound
follows. Integration in $x$ reduces to
$\int_0^1r^{2(\alpha-s)-1}\,dr$, which is finite exactly when $s<\alpha$.
Constants may deteriorate as $\omega\to2\pi$ or $s\to\alpha$.

This threshold concerns a nonzero singular term. Smooth exceptional solutions,
such as the affine half-disk case, need not have that limitation.

## 4. Aubin–Nitsche turns adjoint approximation into an error estimate

### 4.1 The error identity for homogeneous Dirichlet data

Return to $V=H_0^1(\Omega)$ and let $V_h\subset V$ be a conforming Galerkin
space. Put $e=u-u_h$. Define the adjoint solution $z\in V$ by

$$
a(v,z)=(e,v)_{L^2(\Omega)}\qquad\forall v\in V.
$$

The right-hand side is bounded on $V$ by Poincaré; Lax–Milgram therefore gives
$z$. For Poisson the adjoint differential operator is again $-\Delta$ with
homogeneous Dirichlet conditions. Test with $v=e$ and subtract any $z_h\in V_h$:

```{math}
:label: eq:lecture-06-duality-identity
\|e\|_{L^2}^2=a(e,z)=a(e,z-z_h)
\leq |e|_{H^1}\,|z-z_h|_{H^1}.
```

The subtraction is justified by **Galerkin orthogonality**, not by an $L^2$
best-approximation property.

```{prf:theorem} The additional power comes from the adjoint
:label: thm:lecture-06-aubin-nitsche

Assume that there exist $0<t\leq1$, independent-of-$h$ constants
$C_{\rm reg},C_{\rm app}$, and the following two properties:

1. For every $g\in L^2(\Omega)$, the homogeneous adjoint solution $z_g$ satisfies
   $\|z_g\|_{H^{1+t}}\leq C_{\rm reg}\|g\|_{L^2}$.
2. For these solutions,
   $\inf_{v_h\in V_h}|z_g-v_h|_{H^1}\leq
   C_{\rm app}h^t\|z_g\|_{H^{1+t}}$.

Then the conforming, exactly assembled Galerkin solution satisfies

$$
\|u-u_h\|_{L^2}\leq C_{\rm app}C_{\rm reg}h^t|u-u_h|_{H^1}.
$$
```

**Proof.** Apply property 2 in {eq}`eq:lecture-06-duality-identity` and property 1
with $g=e$. This gives $\|e\|_{L^2}^2\leq
C_{\rm app}C_{\rm reg}h^t|e|_{H^1}\|e\|_{L^2}$.
If $e=0$ the result holds; otherwise divide by $\|e\|_{L^2}$. $\square$

For convex polygons and conforming $P_1$ triangles, $t=1$. The $H^2$ regularity
statement in Section 1 and the integer approximation estimate of Lecture 3
verify both hypotheses. Together with the energy estimate, this proves

$$
|u-u_h|_{H^1}\leq Ch\|u\|_{H^2},\qquad
\boxed{\|u-u_h\|_{L^2}\leq Ch^2\|u\|_{H^2}.}
$$

No quasi-uniformity is needed for this upper bound with $h=\max_Kh_K$.
For fixed degree $p\geq1$ and $u\in H^{p+1}$, the same adjoint $H^2$ regularity
adds one power to an energy bound of order $h^p$. Increasing $p$ does not
increase the regularity available from the domain.

### 4.2 Reduced regularity changes both factors

If the primal energy estimate is $|e|_{H^1}\leq Ch^s\|u\|_{H^{1+s}}$ and the
adjoint satisfies the two hypotheses above with exponent $t$, then

$$
\|e\|_{L^2}\leq Ch^{s+t}\|u\|_{H^{1+s}}.
$$

The exponents $s$ and $t$ describe different facts: the actual primal solution
and the solution operator for all adjoint loads in $L^2$. A smooth particular
primal solution does not imply a uniformly smooth adjoint solution operator.
For nonsymmetric forms, use $a(v,z_g)=(g,v)$; regularity of the primal operator
cannot replace the needed regularity of its adjoint.

For a polygon with largest re-entrant angle $\omega$, the Dirichlet regularity
shift is available for $0<t<\pi/\omega$, up to the other corners' thresholds;
see the polygonal regularity discussion in {cite:ts}`BrennerScott2010`.
An affine outer polygon approximating the pacman has convex outer corners.
The fractional approximation estimate in hypothesis 2 is an **additional
approximation result**: Lecture 3 established only the integer version. The
construction and analysis of suitable low-regularity interpolation operators
will be continued in the next lecture. Here the reduced-regularity deduction
is explicitly conditional on that estimate.

In particular, using $s=t<\alpha$ gives an upper bound with every power
$h^{2t}$ below $h^{2\alpha}$. This argument does not prove membership at the
endpoint or an endpoint estimate with an $h$-independent constant. Numerical
slopes close to $\alpha$ and $2\alpha$ are consistent with this picture.

### 4.3 An abstract weaker-norm form

Let $V\hookrightarrow H$ be a **dense continuous embedding of Hilbert spaces**,
and let $a$ be continuous and coercive on $V$, with continuity constant $M$.
Every $g\in H$ defines a bounded functional $v\mapsto(g,v)_H$ on $V$ by the
embedding; no Hahn–Banach extension is needed. Let
$a(v,z_g)=(g,v)_H$. The same orthogonality argument and the dual norm identity
in $H$ give

$$
\|e\|_H\leq M\sup_{0\ne g\in H}
 \frac{\inf_{z_h\in V_h}\|z_g-z_h\|_V}{\|g\|_H}\,\|e\|_V.
$$

The adjoint approximation factor is the entire additional ingredient. In the
Poisson $L^2$ proof, one can choose the single load $g=e$ directly.

## 5. A reproducible pacman convergence experiment

The course repository contains an independent implementation in
`tests/pacman/pacman.cc`.
It retains only the five small initial meshes of the original FSI-suite book
experiment. It uses native deal.II, with no FSI-suite runtime or Gmsh dependency.
The complete regeneration instructions are also provided below.

```{figure} ../assets/lecture-06/pacman-solutions.png
:alt: Five ParaView views of the finite element harmonic solution on uniformly refined pacman meshes with decreasing apertures.
:width: 100%

Computed $P_1$ solutions for $\omega=2\pi,7\pi/4,3\pi/2,5\pi/4,\pi$, after one
global refinement. A shared colour scale is used. The last case is affine.
```

### 5.1 Discretization and the fixed computational domain

Each initial triangulation has a polygonal outer boundary. All manifold ids
are flat and refinement subdivides those triangles without moving the polygon.
The function {eq}`eq:lecture-06-corner` remains harmonic on that fixed domain.
At each level, the discrete boundary values are its nodal interpolant $g_h$.
Thus the experiment includes boundary-data approximation but no changing-domain
geometry error. It uses exact stiffness integration, six levels, and CG with
SSOR to a relative residual tolerance of $5\cdot10^{-13}$.

The limiting $2\pi$ mesh has a slit: opposite banks are distinct, separated by
about $10^{-15}$ in the inherited mesh coordinates. It must not be merged into
an uncut disk. This numerical limit is outside the bounded Lipschitz-domain
hypotheses of the main theorems above. The four other apertures are ordinary
polygonal domains; the half-disk polygon supplies an affine control.

### 5.2 Error integration and independent checks

Errors are integrated against the analytic function and its analytic gradient,
not against a nodal interpolant. The quantities are

$$
E_0=\|u-u_h\|_{L^2},\qquad E_1=|u-u_h|_{H^1},\qquad
\|u-u_h\|_{H^1}=\sqrt{E_0^2+E_1^2}.
$$

A Duffy rule on a triangle maps a square point $(a,b)$ to
$(a^\beta(1-b),a^\beta b)$, with Jacobian $\beta a^{2\beta-1}$.
On cells touching the corner, the collapsed edge is mapped to that vertex and
$\beta=1/\alpha$. The leading squared-gradient singularity then has radial
weight $a^{2\beta\alpha-1}=a$, which improves quadrature accuracy.
Other cells use $\beta=1$. The main rule has eight Gauss points in each square
direction. Reintegration with twelve points on the finest level checks both
norms to a relative tolerance $5\cdot10^{-4}$.

The tests separately check the algebraic residual, the affine solution,
decreasing singular-case errors and broad windows for the final convergence
rates. The residual alone does not validate the assembled PDE, and a colour
plot alone does not establish convergence.

```{figure} ../assets/lecture-06/pacman-cell-errors.png
:alt: ParaView cellwise H1 seminorm errors on the same five pacman meshes, with concentration at the re-entrant corners.
:width: 100%

Each cell is coloured by $\bigl(\int_K|\nabla(u-u_h)|^2\bigr)^{1/2}$, with one
shared scale. This is an integrated exact error, not a pointwise gradient
error or a computable a posteriori estimator. The affine control is nearly zero.
```

### 5.3 Boundary interpolation enters the weaker-norm estimate

The theorem in Section 4 assumed $e\in H_0^1$. Here the trace of the error is
$g-g_h$, so that hypothesis is not automatic. For a sufficiently regular adjoint
$-\Delta z=e$, Green's identity instead gives

$$
\|e\|_{L^2}^2=a(e,z-z_h)
 -\int_{\partial\Omega}(g-g_h)\partial_n z.
$$

Galerkin orthogonality still removes $z_h\in V_{h,0}$. The mismatch vanishes on
the rays and is supported on the outer polygon, away from the singular corner.
On those edges $g$ is smooth and its piecewise linear interpolation has
$\|g-g_h\|_{L^2(\Gamma_{\rm outer})}\leq Ch^2$. If, in addition to adjoint
approximation, the local normal-trace bound
$\|\partial_nz\|_{L^2(\Gamma_{\rm outer})}\leq C\|e\|_{L^2}$ holds, then

$$
\|e\|_{L^2}\leq Ch^t|e|_{H^1}+Ch^2.
$$

This states the extra regularity requirement used to interpret the experiment;
we do not obtain it from the homogeneous-data theorem alone. For the ordinary
pacman polygons the outer corners are convex and separated from the re-entrant
vertex. The special slit case is retained as numerical evidence.

### 5.4 Convergence graphs and tables

With $E_j$ the error at cycle $j$, the measured rate is

$$
p_j=\frac{\log(E_{j-1}/E_j)}{\log(h_{j-1}/h_j)}.
$$

Uniform subdivision halves $h$ here. For quasi-uniform meshes in two dimensions,
$N\asymp h^{-2}$: powers $h^\alpha$ and $h^{2\alpha}$ correspond to slopes
$-\alpha/2$ and $-\alpha$ against DoFs. Distinguish these axes when reading a plot.
The affine case is excluded from slope fits because its error is controlled by
floating-point arithmetic and the linear solver.

```{figure} ../assets/generated/lecture-06/convergence-h1.svg
:alt: H1 seminorm errors against maximum cell diameter and degrees of freedom, for four singular apertures.
:width: 100%

Energy errors. Dashed reference lines have powers $h^\alpha$ and $N^{-\alpha/2}$;
they are anchored at the finest computed point, not fitted to the data.
```

```{figure} ../assets/generated/lecture-06/convergence-l2.svg
:alt: L2 errors against maximum cell diameter and degrees of freedom, for four singular apertures.
:width: 100%

$L^2$ errors. Dashed reference powers are $h^{2\alpha}$ and $N^{-\alpha}$.
The boundary-data approximation and strict regularity thresholds discussed
above remain part of the interpretation.
```

```{include} ../assets/generated/lecture-06/tables.md
```

### 5.5 Regeneration commands

Run from the repository root in the course deal.II environment:

```bash
make pacman
python3 -m pip install numpy matplotlib
make pacman-plots PYTHON=python3
```

`make pacman` builds only the new experiment target in `build/pacman-release`
and runs its five GTest cases. It creates missing CSV, metadata and ParaView
VTU/PVD files under `notes/assets/generated/lecture-06/pacman-0/` through
`pacman-4/`. The plotting command reads those CSV files and creates missing
figures and tables included here. No historical numerical table is used.
Existing complete case outputs are reused and the corresponding numerical tests
are marked skipped. To recompute and validate all cases, use
`make pacman REGENERATE_FIGURES=1`. The plotting and ParaView scripts accept
`--force` to replace existing figures. In CI, `[regenerate figures]` in a commit
message or PR title/body requests full regeneration; otherwise cached outputs
are reused and only missing files are generated.
For a single aperture, after compilation:

```bash
./build/pacman-release/bin/nmpde_pacman_tests \
  --gtest_filter='*ConvergenceAndVtu/2'
```

Open a generated `solution.pvd` file in ParaView. Its time index selects the
refinement cycle. `solution` is $u_h$; `exact_nodal_interpolant` is $I_hu$, not
$u$ between nodes; `nodal_error` is their signed nodal difference;
`cell_H1_error` is the integrated cell error displayed above.

To regenerate both five-panel images without a saved ParaView state:

```bash
pvpython --disable-registry tests/pacman/render_paraview.py
# Or specify a pvpython installation through make:
make pacman-images PVPYTHON=/path/to/pvpython
```

The script uses cycle 1 by default; `--cycle 2 --force` selects a finer level and
`--output output/pacman-images` writes to a separate directory. ParaView is
optional for computing convergence tables. On a headless Linux build that
requires X11, use `xvfb-run -a pvpython --disable-registry ...`.

See `tests/pacman/README.md` in the local checkout for mesh provenance,
checksums, solver settings and the complete output specification.

## 6. Equivalence calculations for independent study

The two calculations below supply the whole-space equivalences mentioned in
Section 3. Their transfer to bounded domains uses the extension theorem stated
there. They concern unrestricted Sobolev spaces, without imposed boundary data.

### 6.1 Fourier and difference quotients

For $v\in\mathcal S(\mathbb R^d)$, change variables $y=x+h$, apply Plancherel to
$v(\cdot+h)-v$, and use Tonelli for the nonnegative integrand:

$$
\begin{aligned}
|v|_{H^s(\mathbb R^d)}^2
&=\int_{\mathbb R^d}\frac{\|v(\cdot+h)-v\|_{L^2}^2}{|h|^{d+2s}}\,dh\\
&=\int_{\mathbb R^d}|\widehat v(\xi)|^2
 \left[\int_{\mathbb R^d}\frac{|e^{i\xi\cdot h}-1|^2}{|h|^{d+2s}}\,dh\right]d\xi\\
&=c_{d,s}\int_{\mathbb R^d}|\xi|^{2s}|\widehat v(\xi)|^2\,d\xi.
\end{aligned}
$$

For the last equality, rotate $\xi/|\xi|$ to a coordinate vector and scale
$h$ by $|\xi|^{-1}$. The constant is positive and finite: near zero,
$|e^{ih_1}-1|^2\leq |h|^2$ gives a radial integral $\int_0^1r^{1-2s}dr$;
at infinity the bound by $4$ gives $\int_1^\infty r^{-1-2s}dr$.
These require respectively $s<1$ and $s>0$.
Since $1+|\xi|^{2s}\asymp(1+|\xi|^2)^s$, adding the $L^2$ norm proves
norm equivalence. Density then extends the identity to the completed spaces.

### 6.2 Fourier and the quadratic interpolation functional

For $(X_0,X_1)=(L^2(\mathbb R^d),H^1(\mathbb R^d))$, put
$w(\xi)=1+|\xi|^2$. Pointwise minimization of
$|\widehat v-\widehat v_1|^2+t^2w|\widehat v_1|^2$ gives

$$
\widehat v_1=\frac{\widehat v}{1+t^2w},\qquad
K_2(t,v)^2=\int\frac{t^2w}{1+t^2w}|\widehat v|^2\,d\xi.
$$

The proposed minimizer lies in $H^1$ for each $t>0$, since
$w/(1+t^2w)^2$ is bounded. The pointwise lower bound and this admissible
minimizer justify minimizing under the integral. Tonelli and $a=t\sqrt w$ yield

$$
\int_0^\infty t^{-2s}K_2(t,v)^2\frac{dt}{t}
=\left(\int_0^\infty\frac{a^{1-2s}}{1+a^2}\,da\right)
 \int_{\mathbb R^d}w^s|\widehat v|^2\,d\xi.
$$

The scalar integral is finite and positive for $0<s<1$. This proves the
whole-space real-interpolation characterization. On a Lipschitz domain,
bounded restriction and a common bounded extension operator on $L^2$ and $H^1$
transfer the equivalence in both directions.

## 7. Exercises

1. For the cutoff corner solution $\chi(r)u$, compute $-\Delta(\chi u)$ and
   identify its support. Explain why it is an $L^2$ forcing term.
2. Let the primal energy error have order $h^{0.8}$ and the adjoint gain be
   $h^{0.6}$. Derive the conditional $L^2$ order. Express both error rates
   as powers of $N$ on quasi-uniform two-dimensional meshes.
3. Regenerate the $3\pi/2$ experiment. Compare rates from the last two levels
   with the fit to the last three. Identify which CSV columns test algebraic
   error and which test error-quadrature sensitivity.
4. Explain why replacing `exact_nodal_interpolant` by the analytic solution
   in an error integral changes the quantity being measured. Explain why the
   affine control should not be assigned a fitted discretization rate.
