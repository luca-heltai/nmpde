# Laboratory 4 — From cell integrals to a Poisson solver

This laboratory turns the construction in [Lecture 5](../lectures/lecture-05.md)
into a working Poisson solver. Three exercises lead from quadrature on physical
cells to the complete assembly, boundary treatment and solve of deal.II
**step-3**. The starter supplies the mesh, class structure, diagnostics and
output. You implement the mathematical operations in the TODOs.

## 1. Objectives and working assumptions

By the end of the laboratory, you should be able to:

- explain how mapping, quadrature and `FEValues` produce a physical integral;
- translate the local Poisson forms into loops over quadrature and basis indices;
- accumulate local contributions through a cell's global DoF indices;
- impose Dirichlet data, solve with CG and verify the result quantitatively.

We use deal.II **9.7.1**, C++17, a serial mesh of uniform squares in
$\Omega=(-1,1)^2$, `FE_Q<2>(1)` and `MappingQ1<2>`. These square-cell mappings
are affine. The mesh is conforming and has no hanging nodes. Every boundary
face has boundary id `0`. The model problem is

$$
-\Delta u=1\quad\text{in }\Omega,\qquad u=0\quad\text{on }\partial\Omega.
$$

Its weak form is: find $u\in H_0^1(\Omega)$ such that

$$
\int_\Omega\nabla u\cdot\nabla v\,dx=\int_\Omega v\,dx
\qquad\text{for all }v\in H_0^1(\Omega).
$$

Use the numbering conventions from [Laboratory 3](lab-03.md): $i,j$ are
cell-local basis indices, $I,J$ are global indices and $g_K(i)$ is a cell's
local-to-global map. Boundary DoFs remain numbered; their values are imposed
after assembly. Interpolation error studies, adaptive refinement, general
parameter files and the dimension-independent step-4 extension are later work.

## 2. Build and run the starter

Work from the repository root in the course deal.II environment:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target lab-04 --parallel 2
mkdir -p build/runs/lab-04-r1
cd build/runs/lab-04-r1
../../bin/lab-04 1 | tee transcript.txt
```

The target builds `labs/lab-04/step-3-starter.cc`. The historical interpolation
source `labs/lab-04/lab-04.cc` is retained separately. Edit the starter to complete
the exercises and rebuild after each stage.

The optional command-line argument is the number $r$ of global refinements,
with $1\le r\le5$; the default is $r=3$. The untouched starter reports 4 active
cells, 9 global DoFs, an unfinished area integral equal to zero and a TODO
message. It writes `mesh.svg` and stops before assembly. A successful initial
run therefore means that the starter builds; the exercises remain to be done.
After all TODOs are complete, it also writes `solution.vtu`.

For a finer mesh, keep its outputs in a separate directory:

```bash
# From build/runs/lab-04-r1, after rebuilding the completed source:
mkdir -p ../lab-04-r3
cd ../lab-04-r3
../../bin/lab-04 3 | tee transcript.txt
```

For this mesh family, with $n=2^r$ subdivisions along each side,

$$
N_{\mathrm{cells}}=n^2,\qquad N_{\mathrm{DoFs}}=(n+1)^2,\qquad
N_D=4n,\qquad N_F=(n-1)^2.
$$

The last two counts are boundary and free DoFs. Derive them before running.
For $r=3$, check 64 cells, 81 DoFs, 32 boundary DoFs and 49 free DoFs.

## 3. The formulas behind the TODOs

The unconstrained conforming space includes boundary basis functions. First
assemble on this whole space; then prescribe the boundary coefficients. For
the restrictions $\varphi_{K,i}$ of the global basis, the local contributions are

$$
A^K_{ij}=\int_K\nabla\varphi_{K,j}\cdot\nabla\varphi_{K,i}\,dx,
\qquad b^K_i=\int_K\varphi_{K,i}\,dx.
$$

With physical quadrature points $x_q=F_K(\widehat x_q)$ and
$W_q=|\det DF_K(\widehat x_q)|w_q$, compute

$$
A^{K,Q}_{ij}=\sum_q\nabla\varphi_{K,j}(x_q)\cdot
\nabla\varphi_{K,i}(x_q)W_q,
\qquad b^{K,Q}_i=\sum_q\varphi_{K,i}(x_q)W_q.
$$

Then accumulate

$$
A_{g_K(i),g_K(j)}\mathrel{+}=A^{K,Q}_{ij},\qquad
b_{g_K(i)}\mathrel{+}=b^{K,Q}_i.
$$

The row corresponds to the test function and the column to the trial function.
Different cells can contribute to the same global entry, so accumulation is
essential. Read Lecture 5, §§5–6, for the reference-cell transformation.

```{figure} ../assets/lecture-05/pipeline.svg
:alt: Reference shape data, mapping and quadrature supply physical values, gradients and weights for local integration, followed by local-to-global assembly.
:name: fig:lab-04-pipeline

The pipeline from Lecture 5 is the implementation path for these exercises.
```

`shape_grad(i,q)` already contains the physical gradient, including the inverse
Jacobian transformation. `JxW(q)` already contains the quadrature weight and
the geometric volume factor. Apply each factor once. On these affine squares,
`QGauss<2>(2)` integrates the $Q_1$ stiffness entries and the constant-load
entries exactly, up to floating-point roundoff. General geometries and data
require their own quadrature analysis.

## 4. Exercise 1 — Integrate over the physical mesh

Complete **TODO 1** in `integrate_area()`.

1. Sketch the reference square and a physical square. For the four-cell mesh,
   determine each cell's side length and area.
2. Construct `FEValues<2>` using the supplied `mapping`, `fe` and `quadrature`.
   Request the update flag needed to access `JxW`.
3. Loop over active cells. Reinitialize `FEValues` on each cell, then sum
   `JxW(q)` over its quadrature points. Add the cell sums to `area`.
4. Rebuild and run with $r=1$. The total must equal $4$ within absolute
   tolerance $10^{-12}$. Print the individual cell sums while checking your
   implementation; compare them with your geometric prediction.
5. Repeat with $r=3$. Explain why the total is unchanged although both the
   cell areas and the number of cells change. Remove verbose per-cell output
   once checked.

**Deliverable:** the two area checks and a short explanation of what `reinit`
changes. The starter now proceeds to the allocation stage. Summing reference
weights alone computes a different quantity: identify the missing factor.

## 5. Exercise 2 — Assemble local contributions into a sparse system

Complete **TODOs 2a, 2b and 2c**, in that order.

### 5.1 Allocate the algebraic objects

In `setup_system()`, create a `DynamicSparsityPattern` with one row and column
per global DoF. Fill it from the `DoFHandler`, copy it into the member
`sparsity_pattern`, and reinitialize `system_matrix`. Size `solution` and
`system_rhs` consistently.

Explain why the sparsity pattern is determined by shared cell supports before
any integral has been computed. A stored position can have numerical value
zero. Keep the member `sparsity_pattern` alive while the matrix uses it.

### 5.2 Compute one cell's matrix and vector

In the supplied active-cell loop of `assemble_system()`:

1. Reinitialize `fe_values` for the current cell.
2. Reset **both** `cell_matrix` and `cell_rhs` to zero on every cell.
3. Implement the quadrature sums in §3 using the supplied local objects.
   The load is $f=1$, so no evaluation of a spatially varying function is
   required. Explain each loop index and each multiplication.
4. On the four-cell mesh, print one local matrix and vector before scattering.
   Check symmetry and verify that the local load sum equals that cell's area.
   Compute one diagonal stiffness entry by hand from a reference $Q_1$ basis
   function and compare it with the program. State the local basis function
   you used.

### 5.3 Accumulate through the local-to-global map

Fill the pre-sized `local_dof_indices` using the cell's `get_dof_indices()`.
Use it to add the local matrix entries and load entries into the global
objects. Preserve the cell-local order. Use `SparseMatrix::add()` for the
matrix and accumulation for the vector.

The supplied `check_assembly()` runs **before** Dirichlet treatment. It checks

$$
A=A^T,\qquad A\mathbf1=0,\qquad \sum_I b_I=|\Omega|=4,
\qquad A_{II}>0.
$$

Explain the first three identities using symmetry, partition of unity and
$\sum_i\nabla\varphi_{K,i}=0$. The code uses a tolerance of $10^{-12}$ scaled
by $\max(1,\|A\|_\infty)$ for the first two checks and an absolute tolerance
of $10^{-12}$ for the load sum. These are roundoff checks for this small mesh
family. They do not establish correctness of every matrix entry; retain your
independent local integral calculation.

**Deliverable:** the local comparison, global diagnostic output and an
explanation of why replacing accumulation by assignment loses contributions.
The program should now stop at TODO 3a. Do not call CG on the unconstrained
matrix: constants are still in its kernel.

## 6. Exercise 3 — Impose Dirichlet data and complete step-3

Complete **TODOs 3a and 3b**.

### 6.1 Prescribe the boundary coefficients

Use `VectorTools::interpolate_boundary_values()` to fill `boundary_values`
for boundary id `0` and `Functions::ZeroFunction<2>()`. Then use
`MatrixTools::apply_boundary_values()` to modify the matrix, right-hand side
and initial solution. Keep column elimination enabled; its default is `true`.
This preserves symmetry for CG.

Relate the operation to the free/boundary partition from Lecture 5, §7:

$$
A_{FF}U_F=b_F-A_{FD}G,\qquad U_D=G.
$$

Here $G=0$. Full Dirichlet data remove the constant kernel, and the free
stiffness matrix is symmetric positive definite. The library keeps a full-size
system with equations prescribing the boundary coefficients. Explain why
setting boundary entries of `solution` alone does not impose these equations.

### 6.2 Solve and check the result

The starter supplies `SolverControl(1000, 1e-12)` and
`SolverCG<Vector<double>>`. Complete the solve call using `system_matrix`,
`solution`, `system_rhs` and `PreconditionIdentity()`. Print the iteration count
after the solve. The solver tolerance is an **absolute Euclidean residual**
tolerance; it is independent of the finite element discretization error.

The supplied solution checks require:

- the expected number of prescribed boundary DoFs;
- $\|b-AU\|_2\le 2\cdot10^{-11}\max(1,\|b\|_2)$ for the **modified** system;
- a maximum boundary coefficient error at most $10^{-12}$;
- for $r=1$, the sole interior coefficient equal to $3/8$ within $10^{-11}$.

Derive the last value independently: sum the four local contributions to the
central diagonal and central load, then solve the single free equation. The
other eight coefficients are zero. This check fixes the scale as well as the
sign of the assembled operator and load.

Run the completed program for $r=1$ and $r=3$, preserving both transcripts.
On each run, the final message must say `Checks passed. Wrote solution.vtu.`
Open `solution.vtu` in ParaView, click **Apply**, select `solution` and use
**Surface With Edges**. Inspect the boundary and interior; optionally apply
**Warp By Scalar**. The picture complements the numerical checks.

**Deliverable:** the completed source, two transcripts, two VTU files and the
hand calculation of the four-cell result. Explain the roles of `make_grid`,
`setup_system`, `assemble_system`, boundary treatment, `solve` and output.
You have now implemented all stages of step-3. Our separate boundary method
makes the pre-Dirichlet diagnostics explicit; the tutorial applies the boundary
data at the end of `assemble_system()`.

## 7. API lookup sheet

The formulas and arguments in this table identify the operations; the TODO
implementations remain your work.

| Operation | deal.II 9.7.1 interface |
|---|---|
| Combine mapping, element and quadrature | `FEValues<2>(mapping, fe, quadrature, flags)` |
| Request geometric weights | `update_JxW_values` |
| Request values or gradients | `update_values`, `update_gradients` |
| Update for one cell | `fe_values.reinit(cell)` |
| Traverse quadrature and local DoFs | `quadrature_point_indices()`, `dof_indices()` |
| Physical value, gradient, weight | `shape_value(i,q)`, `shape_grad(i,q)`, `JxW(q)` |
| Build structural couplings | `DoFTools::make_sparsity_pattern(dof_handler, dsp)` |
| Copy the pattern, initialize the matrix | `sparsity_pattern.copy_from(dsp)`, `system_matrix.reinit(sparsity_pattern)` |
| Size vectors | `vector.reinit(dof_handler.n_dofs())` |
| Obtain $g_K$ | `cell->get_dof_indices(local_dof_indices)` |
| Accumulate one matrix entry | `system_matrix.add(I, J, value)` |
| Interpolate boundary data | `VectorTools::interpolate_boundary_values(dof_handler, boundary_id, function, boundary_values)` |
| Modify equations for Dirichlet data | `MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs)` |
| Solve the algebraic system | `solver.solve(system_matrix, solution, system_rhs, preconditioner)` |

The local index vector must already have `fe.n_dofs_per_cell()` entries when
passed to `get_dof_indices()`. A global DoF index is a
`types::global_dof_index`; it is independent of the index of a mesh vertex.

## 8. Further checks and independent work

- **Affine patch test.** Replace the model by $u=x+y$, $f=0$, with matching
  nonzero Dirichlet data on the same mesh. Adapt the load-sum diagnostic and
  replace the four-cell $3/8$ check. Compare the computed coefficients with
  nodal interpolation of $x+y$ on a mesh with interior DoFs. Require a maximum
  coefficient error at most $10^{-11}$ for $r=1,3$ with the supplied solver
  tolerance. Explain why exact representability and exact integration make
  this a check of assembly and nonzero boundary elimination, without a
  convergence-rate theorem.
- **Quadrature experiment.** On one cell, compare selected local entries from
  one-point and two-point Gauss rules with hand integration. Explain which
  exactness property changes. Complete the main solve with the two-point rule.
- **Error measurements.** Further work with interpolation errors and
  manufactured non-affine solutions can distinguish approximation error,
  quadrature error and solver tolerance. Their convergence analysis connects
  to the next theory lecture.

## 9. Tutorial reference and student source

Use the [version-pinned step-3 source](https://github.com/dealii/dealii/blob/v9.7.1/examples/step-3/step-3.cc)
for a reading comparison after implementing each stage. Match its
`setup_system()`, `assemble_system()`, `solve()` and `output_results()` to your
program and explain the role of every object. The tutorial is a complete
reference program; copying its loops does not replace the exercise checks.

The source below is included directly from the student starter so the page
and the compiled target share the same TODOs.

```{literalinclude} ../../labs/lab-04/step-3-starter.cc
:language: cpp
```
