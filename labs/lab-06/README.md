# Lab 6: Poisson with nonhomogeneous Dirichlet data

Solve a family of Poisson problems on $\Omega=(-1,1)^2$:

$$
-\Delta u=f\quad\text{in }\Omega,\qquad u=g\quad\text{on }\partial\Omega.
$$

A single smooth function supplies the manufactured solution, the forcing
$f=-\Delta u$, the boundary data $g=u|_{\partial\Omega}$, and the reference
for error integration. The mesh is refined uniformly. We use continuous
`FE_Q<2>(1)` elements, with no hanging nodes.

By the end of the exercises you should be able to change the problem data
consistently, verify boundary values, measure convergence in $L^2$ and the
full $H^1$ norm, and inspect a computed field in ParaView.

## 1. Supplied solver and mathematical setting

The starter supplies mesh generation, local and global assembly, symmetric
Dirichlet elimination, CG, residual diagnostics and VTU output. Complete only
the three TODOs in
[`poisson-starter.cc`](../../labs/lab-06/poisson-starter.cc).
The supplied analytical functions are in
[`manufactured_functions.h`](../../labs/lab-06/manufactured_functions.h).
The historical `lab-06.cc` is retained separately; the `lab-06` target now
builds this starter.

The weak formulation is: find $u\in H^1(\Omega)$ with trace $g$ such that

$$
\int_\Omega\nabla u\cdot\nabla v=\int_\Omega fv
\qquad\forall v\in H^1_0(\Omega).
$$

In the discrete space, prescribe the boundary nodal values of $g$. For free
indices $I$ and boundary indices $B$, the interior system is

$$
A_{II}U_I=b_I-A_{IB}U_B,\qquad (U_B)_j=g(a_j).
$$

Explain why setting boundary entries of the solution alone does not produce
this system. `MatrixTools::apply_boundary_values` performs the elimination;
its supplied final argument `true` eliminates columns and preserves symmetry.
For a smooth nonpolynomial $g$, the finite element trace is its nodal
interpolant: agreement at boundary nodes does not imply pointwise agreement
along every boundary edge.

From the repository root, in the course deal.II **9.7.1** environment:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target lab-06 --parallel 2
mkdir -p build/runs/lab-06-affine
cd build/runs/lab-06-affine
../../bin/lab-06
```

Rebuild from the repository root after each source change. Use separate run
directories for the three functions: each run replaces its CSV and VTU files.
The untouched starter runs the homogeneous zero problem and reports a nonzero
boundary error against the intended affine data. It writes only the CSV
header until TODO 3 is completed. A small residual alone therefore does not
validate the intended boundary value problem.

## 2. Exercise 1: An affine solution with nonzero boundary data

Start with the supplied `Lab06::Affine`:

$$
u(x,y)=1+x+y,\qquad f=0,\qquad g=1+x+y.
$$

1. Compute $\nabla u$ and $\Delta u$ by hand. Identify values of $g$ at the four
   corners. Predict whether this function belongs to the discrete space.
2. Complete TODO 1: evaluate minus the Laplacian of `exact` at the physical
   quadrature point. The API is `Function::laplacian(point)`; the selected
   derived function must implement it. The base class does not infer
   derivatives from `value()`.
3. Complete TODO 2: pass the **same `exact` object** to
   `VectorTools::interpolate_boundary_values` in place of `ZeroFunction`.
   Keep the supplied elimination and solver.
4. Run at refinement 2: check **16 cells and 25 DoFs**, a small residual and a
   maximum boundary error below $10^{-12}$. Open `solution-2.vtu` and verify
   the affine profile. Once TODO 3 is complete, both error norms should be
   below $10^{-11}$.

This is a patch test: affine functions belong to $Q_1$, and the forcing and
stiffness integrals here are integrated exactly. Roundoff and solver tolerance
limit the measured error. A convergence slope is not meaningful for this case.

## 3. Exercise 2: A manufactured solution and convergence

Replace the declaration of `exact` with `Functions::ExpFunction<2> exact`.
This function implements

$$
u(x,y)=e^{x+y},\qquad\nabla u=(u,u),\qquad\Delta u=2u.
$$

First derive $f$ and $g$ by hand. Changing the function declaration must now
change forcing, boundary values and error reference together. Keep $p=1$.

Complete TODO 3 using the deal.II error routines introduced in Laboratory 5:

- Allocate a cellwise error vector with `mesh.n_active_cells()` entries.
- Call `VectorTools::integrate_difference` with `dofs`, `solution`, `exact`,
  the cellwise vector, `QGauss<2>(fe.degree+3)` and `VectorTools::L2_norm`.
- Reduce with `VectorTools::compute_global_error`, using the **same norm**.
- Repeat for `VectorTools::H1_norm`. Append `degree,h,l2,h1` to `errors.csv`.

Here $h=2/2^r$ is the cell side length after $r$ global refinements; the cell
diameter is $\sqrt2h$. The full norm is

$$
\|e\|_1=\left(\|e\|_0^2+|e|_1^2\right)^{1/2},
\qquad |e|_1^2=\int_\Omega|\nabla e|^2.
$$

`H1_norm` includes both terms. `H1_seminorm` includes only the gradient term.
These errors compare the **Galerkin solution** $u_h$ with $u$.

Run $r=2,\ldots,6$. There are $4^r$ cells and $(2^r+1)^2$ DoFs. For each pair
of successive meshes compute

$$
q_r=\frac{\log(E_{r-1}/E_r)}{\log(h_{r-1}/h_r)}
    =\log_2(E_{r-1}/E_r).
$$

For this smooth solution on uniform square meshes, nodally interpolated
Dirichlet data and fixed degree $p=1$, expect $\|u-u_h\|_0=O(h^2)$ and
$\|u-u_h\|_1=O(h)$. The constants are independent of $h$; the $L^2$ estimate
uses elliptic regularity on the convex square. The boundary interpolation
must also be accounted for in the approximation estimate.

The assembly uses four Gauss points per direction. Repeat the finest case
with five points in both assembly and error integration. Check that both
errors change by less than $10^{-5}$ relatively. Keep the algebraic residual
small compared with the measured discretization errors.

From the run directory, generate the plot (NumPy and Matplotlib required):

```bash
python ../../../labs/lab-06/plot_results.py errors.csv --output convergence.png
```

The panels show errors against $h$ with slope-2 and slope-1 reference lines.
Read slopes with respect to the horizontal variable: in two dimensions
$N\simeq h^{-2}$, so plotting against DoFs would give exponents $-1$ and
$-1/2$ instead.

### Computed reference results

The figures below are generated from the reference test's computed solution
and errors. They provide graphical comparisons; the exercise implementation
is not displayed here. The test checks the affine patch, residuals, boundary
values, quadrature sensitivity and finest-grid rates (within $0.12$ of 2 and 1).

```{figure} ../../notes/assets/generated/lab-06/exponential-solution.png
:alt: Computed exponential Poisson solution as a surface and a contour view.

Computed $Q_1$ solution at $r=6$. The right panel displays nodal values and
contours; the VTU file contains the finite element field.
```

```{figure} ../../notes/assets/generated/lab-06/exponential-convergence.png
:alt: L2 and full H1 errors for the exponential manufactured solution.

Errors computed with deal.II and reference slopes 2 in $L^2$ and 1 in full $H^1$.
```

## 4. Exercise 3: Two peaks in ParaView

Use `Lab06::TwoPeaks exact`. Its value, gradient and Laplacian are supplied:

$$
u(x,y)=1+0.2x-0.1y+
2e^{-12((x+0.35)^2+(y+0.2)^2)}+
1.2e^{-25((x-0.4)^2+(y-0.35)^2)}.
$$

For a term $w=Ae^{-\alpha|x-c|^2}$ in two dimensions,

$$
\nabla w=-2\alpha(x-c)w,\qquad
\Delta w=(4\alpha^2|x-c|^2-4\alpha)w.
$$

The affine part ensures visibly nonzero boundary data. The narrower peak
requires finer meshes before the asymptotic regime becomes clear.

1. Run at refinements 2 and 5. Open `solution-5.vtu` in ParaView, click
   **Apply**, and color by `solution`. Use **Surface With Edges** to inspect
   the mesh, then **Surface** to inspect the field.
2. Add **Warp By Scalar**, select `solution`, and use scale factor $0.3$.
   Rotate the view to distinguish the two peaks. Add a **Contour** filter
   on the original data if desired.
3. Compare `solution` with `exact_at_nodes`, using the same color range.
   The latter is the nodal interpolant of the exact function, so agreement
   of these displays is a visual check. The error norms integrate against
   the actual analytical function and give a quantitative check.

Keep screenshots of the surface and warped view. Compare the peak locations
and relative heights with the supplied function. Use your error table to
explain what changes under refinement.

```{figure} ../../notes/assets/generated/lab-06/peaks-solution.png
:alt: Computed Poisson solution with two unequal peaks and nonzero boundary data.

Computed two-peak solution, generated from the numerical test at $r=6$.
```

```{figure} ../../notes/assets/generated/lab-06/peaks-convergence.png
:alt: L2 and full H1 error convergence for the two-peak problem.

Uniform-refinement errors for the two-peak solution. The finest meshes
resolve the narrower peak and approach the predicted orders.
```

## 5. Deliverables and optional extension

Submit the completed starter, the exponential error table and plot, and two
ParaView screenshots of the two-peak solution. Report the affine patch errors,
the finest observed rates and the quadrature comparison. Explain how one
function supplies all three roles: forcing, boundary data and exact reference.

As an optional extension, repeat the exponential experiment with degree 2.
Expect orders 3 and 2; adapt the plot's reference slopes accordingly. Keep
this extension separate from the required degree-1 experiments.
