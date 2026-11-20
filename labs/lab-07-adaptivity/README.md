# Lab 7 — Hanging nodes and cell-residual marking

## Problem and uniform-refinement baseline

Solve $-\Delta u=f$ on $(0,1)^2$ with Dirichlet data $g=u|_{\partial\Omega}$.
The starter uses continuous $Q_1$ elements on axis-aligned squares, assembled
cell by cell as in step-4. It imposes $g$ with
`MatrixTools::apply_boundary_values` and refines globally.

The supplied parameter file uses

$$
u(x,y)=x+2y+\sin(\pi x)\sin(\pi y),
\qquad f(x,y)=2\pi^2\sin(\pi x)\sin(\pi y).
$$

The affine term gives nonzero boundary data. If you change **Exact solution**
in `poisson.prm`, derive and set **Right hand side** to $-\Delta u$.
The code takes the boundary data from the exact solution.

Build and run from the repository root. Use a separate working directory
so each experiment has its own output.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target lab-07-adaptivity --parallel 2

repo="$PWD"
run_dir="$(mktemp -d)"
cp labs/lab-07-adaptivity/poisson.prm "$run_dir/"
cd "$run_dir"
"$repo/build/bin/lab-07-adaptivity" poisson.prm | tee uniform.txt
```

`ParsedConvergenceTable` records one row after each solve and prints the
table at the end. The default run should give approximately

| Active cells | DoFs | $\|u-u_h\|_0$ | $\lvert u-u_h\rvert_1$ |
| ---: | ---: | ---: | ---: |
| 16 | 25 | $3.0392\cdot10^{-2}$ | $5.0137\cdot10^{-1}$ |
| 64 | 81 | $7.6010\cdot10^{-3}$ | $2.5151\cdot10^{-1}$ |
| 256 | 289 | $1.9006\cdot10^{-3}$ | $1.2587\cdot10^{-1}$ |
| 1024 | 1089 | $4.7517\cdot10^{-4}$ | $6.2952\cdot10^{-2}$ |

Check the expected orders in $h$ by taking the base-two logarithm of
successive error ratios. Uniform refinement halves $h$, so the expected
orders are 2 for $\|u-u_h\|_0$ and 1 for $\lvert u-u_h\rvert_1$. The
table's rate columns use the DoF counts, which only asymptotically
quadruple at each cycle.

## Exercise 1 — Dirichlet values become affine constraints

The slides introduce the affine relation
$U_i=\sum_j C_{ij}U_j+b_i$ for a constrained coefficient. Prescribing a
Dirichlet value is the special case $U_i=g(a_i)$ with no free coefficients
on the right-hand side. With $U=P\widehat U+b$, the Galerkin system in the
free coefficients uses $P^\mathsf{T}AP$ and $P^\mathsf{T}(F-Ab)$.
The $-Ab$ term accounts for the inhomogeneous data.

Keep uniform refinement and change the way the algebraic system is built.

1. Add an `AffineConstraints<double>` object to `Poisson`. In
   `setup_system()`, clear it after distributing the DoFs. Interpolate
   the exact solution on boundary id 0 using the
   `VectorTools::interpolate_boundary_values` overload that fills the
   constraints. Close the object after all relations have been entered.
2. Pass the closed constraints to `DoFTools::make_sparsity_pattern`.
   Keep the quadrature and the local integrals in `assemble_system()`.
   Replace the manual scatter by the overload of
   `distribute_local_to_global` that takes both the cell matrix and
   the cell right-hand side. Remove the call to
   `MatrixTools::apply_boundary_values`.
3. After solving, call `constraints.distribute(solution)` before
   computing errors. This reconstructs the constrained coefficients.

Compare all four rows with `uniform.txt` at the printed precision.
For an additional check, set $u=x+2y$ and $f=0$. On the uniform mesh the
solution lies in $Q_1$. Check that $\|u-u_h\|_0<10^{-10}$ and
$\lvert u-u_h\rvert_1<10^{-6}$. The seminorm check allows for the
finite-difference gradient of `FunctionParser`. Restore the original
parameters before Exercise 3.

## Exercise 2 — Constraints restore continuity at hanging nodes

A hanging node is a support point carried only by the fine side of a
coarse–fine interface. Continuity determines its value from the coarse
trace. For the $Q_1$ mesh here,
such a point is a fine-cell vertex in the interior of a coarse edge.
The fine side has a midpoint coefficient $U_M$ and the coarse side
has a single linear trace determined by the endpoint coefficients
$U_A,U_B$. Matching the two traces requires

$$
U_M=\frac{U_A+U_B}{2}.
$$

Each fine trace is linear on its half-edge, so equality at $A,M,B$ is
enough for continuity along the entire interface. For higher degree,
the general rule is to evaluate the coarse polynomial trace at the
fine-side support points.

```{figure} ../assets/lab-07/hanging-node.svg
:alt: A coarse square meets two fine squares at an edge with midpoint M. A graph shows the fine and coarse traces agreeing at the endpoints and disagreeing at M before the constraint.

The dark blue curve is the coarse trace. Once the orange midpoint value is
set to the mean of the endpoint values, both orange segments lie on the
blue line.
```

```{figure} ../assets/lab-07/slide19-mesh.png
:alt: A numbered square mesh with the bottom-left quadrant refined; nodes 12 and 13 are hanging.

The numbered mesh from Luca Heltai's *Lab — Constraints*, p. 19.
Identify the coarse edge for node 12 and for node 13. Write both affine
relations using the endpoint numbers before calling the library routine.
```

```{figure} ../assets/lab-07/slide19-basis.png
:alt: Three plots from slide 19 of unconstrained basis functions on a coarse–fine interface.

From top to bottom, the unconstrained nodal functions $N_0$, $N_2$ and
$N_{12}$ from the same slide. A general combination of the unconstrained
functions may have a jump across the interface. The relations above
select the continuous combinations. Set the free coefficient at node 2
to one and the other free coefficients to zero. Use your relations to
express the resulting continuous basis function in terms of $N_2$,
$N_{12}$ and $N_{13}$.
```

1. Start from four coarse squares with `Initial refinement = 1` and
   `Number of cycles = 2`. In `refine_grid()`, use `cell->set_refine_flag()`
   to mark the bottom-left active cell, identified by its center.
   Call `triangulation.execute_coarsening_and_refinement()` after marking.
   The second mesh should have seven active cells.
2. After every `distribute_dofs()`, clear and rebuild the constraints
   for the current mesh. Call `DoFTools::make_hanging_node_constraints`
   before inserting the Dirichlet constraints and before `close()`.
   Keep the same constrained scatter and the same post-solve
   distribution as in Exercise 1.
3. On the second mesh inspect the two hanging-node relations before
   adding boundary values and calling `close()`. The closed constraints
   can absorb a prescribed endpoint value into the inhomogeneous term.
   Use the actual DoF indices, which can differ from those in the slides,
   to verify both original midpoint identities after
   `distribute(solution)`, to within $10^{-12}$ for the supplied data.
   With $u=x+2y$ and $f=0$, check both errors using the tolerances in
   Exercise 1. This affine patch test exercises the boundary and
   hanging-node constraints together.
   Restore the original data afterward.

## Exercise 3 — The cell residual drives two marking rules

On these axis-aligned squares a $Q_1$ field is of the form
$a+bx+cy+dxy$ inside each cell, so $\Delta(u_h|_T)=0$. Compute

$$
q_T=\int_T f^2\,dx,qquad
m_T=\sqrt{q_T}=\|f\|_{0,T},\qquad
e_{T,\mathrm{cell}}=h_Tm_T,\qquad h_T=\operatorname{diam}T.
$$

Here $e_{T,\mathrm{cell}}$ is the cell part of $e_T$ from
[Lecture 8](../lectures/lecture-08.md). This exercise omits the
flux-jump part, so the global reliability result for the full residual
estimator does not apply to this indicator alone.

1. In a loop over active cells, use `FEValues` with
   `update_quadrature_points | update_JxW_values`. Accumulate $q_T$
   from `forcing.value(x_q)` and `JxW(q)`. Store the contributions
   by `cell->active_cell_index()` and sum them on the current mesh.
   Compute $e_{T,\mathrm{cell}}$ for each cell. For the supplied
   manufactured solution, verify $\sqrt{\sum_T q_T}\approx\pi^2$.
2. Replace the single-cell mark from Exercise 2 by the two rules below.
   Each rule marks using the forcing alone. Mark all qualifying cells
   before executing refinement and rebuild the DoFs and constraints
   on the resulting mesh.

   | Marking rule | Mark $T$ when |
   | --- | --- |
   | Global norm divided by the number of active cells $N$ | $m_T>\sqrt{\sum_K q_K}/N$ |
   | Mean squared contribution | $q_T>\sum_K q_K/N$ |

   Express the second threshold in terms of $m_T$ and explain why the
   first is smaller by a factor $\sqrt N$ for $N>1$ and $f\ne 0$.
   In exact arithmetic, predict which cells each rule would mark for a
   nonzero constant forcing on a uniform mesh. In code, report the number marked
   before refinement. If no cell is marked, stop that run instead of
   appending repeated rows for an unchanged mesh.
3. Run both rules starting from the same mesh as the uniform baseline
   and with the same manufactured solution. For the adaptive runs
   set `Rate mode = none`. Save the final `ParsedConvergenceTable`
   for each run. Report at least the active cell count, the DoF count,
   $\|u-u_h\|_0$ and $\lvert u-u_h\rvert_1$ per cycle.
   Compare errors at comparable DoF counts and explain any differences
   in the marked fractions. The number of cycles alone does not give
   the same comparison across the three meshes.

The adaptive loop is solve, record the error, estimate, mark, then refine
before the next solve. The last row refers to the last mesh actually solved.
