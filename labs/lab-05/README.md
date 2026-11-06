# Lab 5: Interpolation errors and convergence

One exercise leads from a known smooth function to a convergence plot. You
will construct its finite element interpolant, evaluate it locally by hand,
integrate its error in $L^2$, and compare your calculations with deal.II.
The final step computes the error in the full $H^1$ norm.

## 1. Function, mesh and expected convergence

Use the square $\Omega=(-1,1)^2$, uniform square meshes, `MappingQ1<2>` and
continuous tensor-product elements `FE_Q<2>(p)`, with $p=1,2,3$.
The exact function is `Functions::CosineFunction<2>`:

$$
u(x,y)=\cos(\pi x/2)\cos(\pi y/2).
$$

It is smooth and vanishes on the boundary. There is no differential equation
to solve in this exercise. The vector represents the nodal interpolant
$I_hu$, determined by $(I_hu)(a_I)=u(a_I)$ at every global support point $a_I$.
The mesh has no hanging nodes, so no constraints are needed.

After $r$ global refinements, the cell side length is $h=2/2^r$,
the number of cells is $4^r$, and the number of DoFs is $(p\,2^r+1)^2$.
The cell diameter is $\sqrt{2}h$; this constant factor does not change slopes.

Recall the reference-element interpolation and scaling argument from
[Lecture 3](../../notes/lectures/lecture-03.md). On this family of affine
square meshes the tensor-product nodal interpolant reproduces $Q_p$, which
contains all polynomials of total degree at most $p$. The same argument gives,
for a smooth function and **each fixed degree** $p$,

$$
\|u-I_hu\|_{L^2(\Omega)}\le C_p h^{p+1}|u|_{H^{p+1}(\Omega)},
\qquad
|u-I_hu|_{H^1(\Omega)}\le C_p h^p|u|_{H^{p+1}(\Omega)}.
$$

The constants are independent of $h$ and can depend on $p$ and the reference
nodes. Nodal values are well defined here because $u$ is smooth. Conformity
allows us to sum the cell gradient integrals to obtain the global seminorm.
Combining both estimates gives an order-$h^p$ bound for the full $H^1$ norm
on the meshes used below ($h\le1$).

Thus we expect slopes $p+1$ in $L^2$ and $p$ in $H^1$ as $h\to0$.
These are upper bounds; the experiment checks whether the predicted orders
are observed for this function. They do not prove a uniform-in-$p$ convergence
rate when only the polynomial degree is increased on a fixed mesh.

## 2. Build and run the student starter

The root target `lab-05` builds
[`interpolation-starter.cc`](../../labs/lab-05/interpolation-starter.cc).
The historical Poisson implementation remains in `lab-05.cc` and is not the
source of this target. Use the course deal.II **9.7.1** environment.

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target lab-05 --parallel 2
mkdir -p build/runs/lab-05
cd build/runs/lab-05
../../bin/lab-05
```

The untouched starter compiles and stops with a `TODO 1` message and exit
code 1. It writes only the CSV header. Complete each TODO and remove its
associated exception. Rebuild from the repository root after editing.
Run in this dedicated directory: each run replaces `errors.csv` there.

First work with the supplied single case $p=1$, $r=2$. Check 16 cells and 25
DoFs. Keep the loops over degree and refinement for point 5.

## 3. A single exercise in six steps

### Step 1. Construct the nodal interpolant

The starter supplies the mesh, finite element, `DoFHandler`, exact function
and a global coefficient vector `interpolated` of the correct size.
Complete TODO 1 using:

```cpp
VectorTools::interpolate(dof_handler, exact, interpolated);
```

This assigns $U_I=u(a_I)$ to every nodal coefficient. The coefficients are
values at support points; quadrature points generally lie elsewhere.

### Step 2. Evaluate the local interpolant by hand

For a cell $K$, write $g_K(i)$ for its local-to-global DoF map. Its local
interpolating polynomial, evaluated at a physical quadrature point $x_q$, is

$$
(I_hu)|_K(x_q)=\sum_{i=0}^{n_K-1} U_{g_K(i)}\varphi_{K,i}(x_q).
$$

Complete TODO 2 with two loops: one over quadrature points and one over local
basis functions. Accumulate the sum in `manual_values[q]`.

Use these objects already supplied by the starter:

- `cell->get_dof_indices(indices)` gives $g_K$;
- `interpolated[indices[i]]` gives $U_{g_K(i)}$;
- `fe_values.shape_value(i,q)` gives $\varphi_{K,i}(x_q)$ after `reinit(cell)`.

Initialize every sum to zero for each cell. Explain why using
`interpolated[i]` in this formula would generally be incorrect. You are
evaluating a polynomial from its nodal coefficients, so calling
`exact.value(x_q)` cannot replace this local calculation.

### Step 3. Compare with FEValues::get_function_values

After the manual evaluation works, complete TODO 3 using:

```cpp
fe_values.get_function_values(interpolated, library_values);
```

This performs the local-to-global lookup and basis-function summation from
step 2. The input is the global coefficient vector; the output has one value
per quadrature point on the current cell. Call it again after each cell's
`reinit`.

Compare `manual_values[q]` and `library_values[q]` at **every quadrature point
of every cell**. Update `max_value_difference` with the largest absolute
difference. For this function, whose magnitude is at most one, require
`max_value_difference < 1e-12`. Report the maximum to the terminal.
This checks two evaluations of the same interpolant; it does not measure its
approximation error relative to $u$.

### Step 4. Integrate the L2 error by hand

The error to compute is

$$
E_{L^2}=\left(\sum_K\int_K |u(x)-I_hu(x)|^2\,dx\right)^{1/2}.
$$

Quadrature gives

$$
E_{L^2}^2\approx\sum_K\sum_q
\bigl[u(x_q)-\texttt{manual\_values}[q]\bigr]^2
\underbrace{w_q|\det DF_K(\widehat x_q)|}_{\texttt{fe\_values.JxW(q)}}.
$$

Complete TODO 4. Obtain $x_q$ with `fe_values.quadrature_point(q)` and evaluate
$u(x_q)$ with `exact.value`. Accumulate **squared** differences times `JxW`
over every cell. Take the square root only after the global sum.

The starter requests `update_values`, `update_quadrature_points` and
`update_JxW_values`; connect each flag to one quantity in the formula.
Do not multiply by the cell area again: `JxW` already contains the Jacobian.
The Euclidean norm of the coefficient vector is not the spatial $L^2$ norm.

Use `QGauss<2>(p+3)` initially. The squared error is not polynomial because it
contains the cosine. Repeat a case with `QGauss<2>(p+5)` and check that the
relative change in the reported error is below $10^{-6}$, away from roundoff.
If it is larger, increase the quadrature order before interpreting slopes.

### Step 5. Write a CSV and produce the convergence plot

Replace the single case in `main` with loops over $p=1,2,3$ and
$r=1,\ldots,6$. Each call constructs a fresh mesh and finite element. The
starter already writes a CSV header and one row per completed case:

```text
degree,refinement,h,dofs,l2
```

Use ordinary stream output with the supplied precision. You do not need an
error-table class. Check the DoF counts against $(p\,2^r+1)^2$.

The provided [`plot_convergence.py`](../../labs/lab-05/plot_convergence.py) uses Matplotlib and
groups the rows by degree. From `build/runs/lab-05`, run:

```bash
python3 ../../../labs/lab-05/plot_convergence.py errors.csv
```

It writes `convergence.png`: a log–log plot of $E_{L^2}$ against $h$, with one
curve per degree and dashed lines of slopes $p+1$. It also prints the observed
rate from the finest two meshes:

$$
\mathrm{rate}=\frac{\log(E(h)/E(h/2))}{\log 2}.
$$

Expect rates approaching 2, 3 and 4 for $p=1,2,3$. Explain why halving $h$
should approximately divide the error by $2^{p+1}$ in the asymptotic regime.
Compare several refinements and check quadrature before interpreting a
discrepancy. Very small errors can be affected by roundoff.

### Step 6. Use deal.II error integration and compute the H1 norm

After the manual calculation and the first plot, use the following facilities
at the end of `run_case`:

```cpp
Vector<double> cell_errors(triangulation.n_active_cells());
VectorTools::integrate_difference(mapping,
                                 dof_handler,
                                 interpolated,
                                 exact,
                                 cell_errors,
                                 quadrature,
                                 VectorTools::L2_norm);
const double library_l2 =
  VectorTools::compute_global_error(triangulation,
                                   cell_errors,
                                   VectorTools::L2_norm);
```

`integrate_difference` returns one cell error norm per active cell.
`compute_global_error` combines them into the global norm. In $L^2$ it takes
the square root of the sum of the **squares** of these cell norms.

Compare `library_l2` with your manual error using the same mapping,
interpolant and quadrature. Require a relative difference below $10^{-10}$
for the proposed cases, where the error is well above roundoff. Agreement
checks the manual integration; the separate quadrature check is still needed.

Now compute the **full** $H^1$ error:

$$
E_{H^1}=\left(\int_\Omega |u-I_hu|^2
                  +|\nabla u-\nabla I_hu|^2\,dx\right)^{1/2}.
$$

Make a second call to `integrate_difference` with `VectorTools::H1_norm`,
then combine the cell errors with `compute_global_error` using the same norm.
`CosineFunction` already provides the exact gradient. The error-integration
routine creates the required evaluation objects internally.

The option `H1_seminorm` would compute only the gradient term. Keep the full
norm requested here and check $E_{H^1}\ge E_{L^2}$, up to roundoff.

Append an `h1` column to both the CSV header and every data row. Rerun the
experiment and the Python script: it detects the column and adds an $H^1$
panel with reference slopes $p$. If needed, finish this final comparison
after class.

## 4. Deliverables and checks

Submit your completed source, `errors.csv` and `convergence.png`, with a short
comment reporting:

- the maximum difference between manual and `get_function_values` evaluations;
- agreement of the manual and deal.II $L^2$ errors and the quadrature check;
- the observed $L^2$ rates for all three degrees;
- the full $H^1$ errors and their rates when the final step is complete.

A graph alone is insufficient: include the numerical checks and explain the
predicted slopes using the interpolation estimates.
