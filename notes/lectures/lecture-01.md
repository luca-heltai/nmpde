# Lecture 1 — Model problems and the Ritz–Galerkin method

This lecture introduces the variational point of view that will be used throughout
the course. We start from a model elliptic problem, derive its weak formulation,
and then restrict the problem to a finite-dimensional space. The main results are
the Lax–Milgram lemma, Galerkin orthogonality, and Céa's lemma.

## 1. A model problem

Let \(\Omega \subset \mathbb{R}^d\) be a bounded domain with sufficiently regular
boundary \(\partial\Omega\). We consider the Poisson problem with homogeneous
Dirichlet boundary conditions:

```{math}
:label: eq:lecture-01-poisson-strong
-\Delta u = f \quad \text{in } \Omega, \qquad
u = 0 \quad \text{on } \partial\Omega.
```

Here \(f\) is the given source term and \(u\) is the unknown solution. This is a
useful model problem because it contains the main ingredients of elliptic finite
element methods while remaining simple enough to analyze explicitly.

There are two broad ways to obtain a numerical approximation:

- **discretize the differential operator directly**, as in finite-difference methods;
- **keep the differential operator in a weak form and approximate the space** in
  which the solution is sought, as in finite-element methods.

The second point of view is particularly useful on non-rectangular domains and
when the data or the solution are not smooth enough for the strong equation to be
interpreted pointwise.

## 2. The weak formulation

Let \(v\) be a smooth test function that vanishes on the boundary. Multiplying
{eq}`eq:lecture-01-poisson-strong` by \(v\), integrating over \(\Omega\), and
integrating by parts gives

```{math}
\int_\Omega \nabla u \cdot \nabla v\,\mathrm{d}x
= \int_\Omega f v\,\mathrm{d}x.
```

The boundary term vanishes because both \(u\) and \(v\) have zero trace. The
natural space for this problem is

```{math}
V = H_0^1(\Omega),
```

and the weak problem is:

```{math}
:label: eq:lecture-01-weak
\text{Find } u \in V \text{ such that }\quad
a(u,v) = \langle f,v\rangle \quad \forall v \in V,
```

where

```{math}
:label: eq:lecture-01-bilinear-form
a(u,v) := \int_\Omega \nabla u \cdot \nabla v\,\mathrm{d}x.
```

The right-hand side is written as a duality pairing because, in general, the data
may belong to \(V'\), the dual of \(V\), rather than to \(L^2(\Omega)\).

## 3. Well-posedness and the Lax–Milgram lemma

The abstract form of the problem is determined by a bilinear form \(a:V\times
V\to\mathbb{R}\) and a functional \(f\in V'\). Two properties are central:

- **boundedness:** there is a constant \(M>0\) such that
  \(\lvert a(w,v)\rvert \leq M\lVert w\rVert_V\lVert v\rVert_V\) for all
  \(w,v\in V\);
- **coercivity:** there is a constant \(\alpha>0\) such that
  \(a(v,v)\geq\alpha\lVert v\rVert_V^2\) for all \(v\in V\).

```{prf:theorem} Lax–Milgram lemma
:label: thm:lecture-01-lax-milgram

Let \(V\) be a Hilbert space and let \(a:V\times V\to\mathbb{R}\) be bounded
and coercive. For every \(f\in V'\), there exists a unique \(u\in V\) such
that

```{math}
a(u,v)=\langle f,v\rangle \quad \forall v\in V.
```

Moreover,

```{math}
\lVert u\rVert_V \leq \frac{1}{\alpha}\lVert f\rVert_{V'}.
```
```

For the Poisson problem, boundedness follows from the Cauchy–Schwarz inequality:

```{math}
\lvert a(u,v)\rvert
\leq \lVert\nabla u\rVert_{L^2(\Omega)}
          \lVert\nabla v\rVert_{L^2(\Omega)}.
```

On \(H_0^1(\Omega)\), Poincaré's inequality makes the seminorm
\(\lVert\nabla v\rVert_{L^2(\Omega)}\) equivalent to the \(H^1\)-norm. Hence
the form is also coercive, and the weak Poisson problem has a unique solution.

## 4. The Ritz–Galerkin approximation

Let \(V_h\subset V\) be a finite-dimensional conforming space. The Galerkin
approximation is obtained by restricting the weak problem to \(V_h\):

```{math}
:label: eq:lecture-01-discrete
\text{Find } u_h\in V_h \text{ such that }\quad
a(u_h,v_h)=\langle f,v_h\rangle \quad \forall v_h\in V_h.
```

Because the same boundedness and coercivity estimates hold on the subspace
\(V_h\), the discrete problem is also uniquely solvable.

Choose a basis \(\{\varphi_1,\ldots,\varphi_N\}\) of \(V_h\) and write

```{math}
u_h = \sum_{j=1}^N U_j\varphi_j.
```

Testing {eq}`eq:lecture-01-discrete` with each basis function gives the linear
system

```{math}
:label: eq:lecture-01-linear-system
\sum_{j=1}^N a(\varphi_j,\varphi_i)U_j
= \langle f,\varphi_i\rangle, \qquad i=1,\ldots,N.
```

Thus the stiffness matrix and load vector are

```{math}
A_{ij}=a(\varphi_j,\varphi_i),
\qquad F_i=\langle f,\varphi_i\rangle,
\qquad A\mathbf{U}=\mathbf{F}.
```

When \(a\) is symmetric, as for the Poisson problem, \(A\) is symmetric positive
definite. This is the algebraic form of the Ritz–Galerkin method.

### The Ritz interpretation

For a symmetric bilinear form, define the energy functional

```{math}
:label: eq:lecture-01-energy
J(w) := \frac{1}{2}a(w,w)-\langle f,w\rangle.
```

The solution of the continuous problem is the unique minimizer of \(J\) over
\(V\), while \(u_h\) is the unique minimizer over \(V_h\):

```{math}
u = \operatorname*{arg\,min}_{w\in V}J(w),
\qquad
u_h = \operatorname*{arg\,min}_{w_h\in V_h}J(w_h).
```

The term *Ritz* emphasizes this minimization property; *Galerkin* emphasizes the
orthogonality condition that characterizes the discrete solution.

## 5. Galerkin orthogonality

The exact solution also satisfies the weak equation for every \(v_h\in V_h\),
since \(V_h\subset V\). Subtracting the discrete equation gives

```{math}
:label: eq:lecture-01-orthogonality
a(u-u_h,v_h)=0 \quad \forall v_h\in V_h.
```

This is **Galerkin orthogonality**: the error \(u-u_h\) is orthogonal to the
discrete space with respect to the bilinear form \(a\). It is the key identity
behind the basic a priori error estimate.

## 6. Céa's lemma

```{prf:theorem} Céa's lemma
:label: thm:lecture-01-cea

Assume that \(a\) is bounded with constant \(M\) and coercive with constant
\(\alpha\). Let \(u\in V\) solve the continuous problem and let \(u_h\in V_h\)
solve {eq}`eq:lecture-01-discrete`. Then

```{math}
\lVert u-u_h\rVert_V
\leq \frac{M}{\alpha}
\inf_{v_h\in V_h}\lVert u-v_h\rVert_V.
```

```

The estimate separates the analysis into two parts:

1. \(M/\alpha\) measures the stability of the variational problem;
2. the infimum measures the best approximation available in \(V_h\).

To see the mechanism, take any \(v_h\in V_h\). By coercivity and
Galerkin orthogonality,

```{math}
\begin{aligned}
\alpha\lVert u-u_h\rVert_V^2
&\leq a(u-u_h,u-u_h)\\
&=a(u-u_h,u-v_h)+a(u-u_h,v_h-u_h)\\
&=a(u-u_h,u-v_h)\\
&\leq M\lVert u-u_h\rVert_V\lVert u-v_h\rVert_V.
\end{aligned}
```

After cancelling the error norm and taking the infimum over \(v_h\), we obtain
Céa's estimate.

For finite elements, this result reduces the error analysis to an approximation
question: how well can the exact solution be approximated by the chosen finite
element space? The following lectures will address the construction of these
spaces and the estimates needed to answer that question.

## Summary

- The weak formulation replaces the strong differential equation with an
  equation in a function space.
- Lax–Milgram gives existence, uniqueness, and stability under boundedness and
  coercivity.
- The Ritz–Galerkin method restricts the variational problem to a finite-
  dimensional conforming space.
- Galerkin orthogonality is the fundamental error identity.
- Céa's lemma bounds the numerical error by the best approximation error in the
  discrete space.
