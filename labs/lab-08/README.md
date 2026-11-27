# Lab 8 — Boundary integrals, flux jumps, and residual estimation

Extend the adaptive Poisson solver from [Lab 7](lab-07-adaptivity.md).
The starter already includes affine Dirichlet constraints, hanging-node
constraints, the cell residual, both marking rules, and the error table.
It uses continuous $Q_1$ elements on axis-aligned squares. The new work
takes place on faces.

## Build and reproduce the Lab 7 baseline

From the repository root, build the target in the course environment:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target lab-08 --parallel 2

repo="$PWD"
lab="$repo/labs/lab-08"
exe="$repo/build/bin/lab-08"
run_dir="$(mktemp -d)"
cd "$run_dir"
mkdir baseline neumann estimator
(cd baseline && "$exe" "$lab/poisson.prm")
```

With the supplied **mean-square** marking rule, the first mesh has 16
cells and the solver marks 4. The sequence of active cell counts is
16, 28, 88, 232. To repeat the other Lab 7 runs, copy the parameter
file and set `Refinement` to `norm-over-cells` or `uniform`.
The program prints the errors and writes a `solution.pvd` series with
the discrete solution and the cell field `indicator`. The reported
`eta` is the Euclidean norm of the local indicators. In the baseline,
it contains the cell contributions alone.

## Exercise 1 — A Neumann face contributes to the load vector

Use the unit square with mixed boundary conditions,

$$
-\Delta u=f,\qquad u=g_D\text{ on }\Gamma_D,\qquad
\nabla u\cdot n=g_N\text{ on }\Gamma_N.
$$

Here $n$ is the outward normal. The starter assigns id 1 to the
right edge when `Mixed boundary = true`. The remaining edges have
id 0 and carry Dirichlet data from the exact solution.

```{figure} ../assets/lab-08/neumann-boundary.svg
:alt: On the unit square, the left, top and bottom edges are solid and marked id 0. The dashed right edge is id 1, and its outward normal points right.

Boundary ids for the mixed problem. Refined boundary faces inherit the
id of their parent face.
```

1. Integrate by parts against a test function $v$ with zero trace on
   $\Gamma_D$. Determine the sign and the destination of the face
   integral. The weak right-hand side is
   $(f,v)_\Omega+\int_{\Gamma_N}g_Nv$.
2. At **TODO 1** in `assemble_system()`, build a `FEFaceValues<2>`
   using `QGauss<1>`, `update_values`, `update_quadrature_points`, and
   `update_JxW_values`. For each local face, check `at_boundary()`
   and `boundary_id() == neumann_id`. Call `reinit(cell, face_no)`
   and add the contribution
   $g_N(x_q)\varphi_i(x_q)\operatorname{JxW}(q)$ to `cell_rhs(i)`
   before scattering the local system.
3. Run the check below. For $u=x(1+y)$, derive $f=0$ and $g_N=1+y$
   on the right edge. This polynomial belongs to $Q_1$. Expect
   $\|u-u_h\|_0<10^{-10}$ and $|u-u_h|_1<10^{-6}$.
   The second tolerance allows for the numerical derivative used
   by `FunctionParser` in the error calculation.

```bash
(cd neumann && "$exe" "$lab/neumann.prm")
```

## Exercise 2 — Two cells supply the traces on an interface

The diagnostic interpolates $v_h=2|x-1|$ on $[0,2]\times[0,1]$. The
interface is $x=1$. The field is continuous there and has a constant
gradient on each side. Take the normal $n$ outward from the first
cell passed to `reinit` and define

$$
j_F=(\nabla v_h|_{\mathrm{first}}-\nabla v_h|_{\mathrm{second}})\cdot n.
$$

```{figure} ../assets/lab-08/interface-quadrature.svg
:alt: Left panel shows a whole face shared by two equal cells with two Gauss points. Right panel shows the right square split into four cells. On x equals 1, the lower fine cell matches subface 0 of the left face and the upper fine cell matches subface 1. Each half has its own two Gauss points and the halves meet at the hanging node.

The drawings show the same rectangle before and after refining its right
square. Each quadrature point is evaluated from both adjacent cells.
```

1. **Whole face on both sides.** Compute the two gradients, the
   normal from the right cell, and $\int_F j_F^2\,ds$ by hand.
   Read the six arguments of `FEInterfaceValues::reinit` in
   `inspect_interface()`. Each side is a triple
   `(cell, face number, subface number)`. `whole_face` is
   `numbers::invalid_unsigned_int`, the marker for a whole face.
   Complete **TODO 2** in `squared_normal_jump()`. Use
   `get_jump_in_function_gradients(solution, jumps)`, then
   `normal(q)` and `JxW(q)`. Verify the result with the first command
   below.
2. **Whole face and subface.** For each of the two fine cells at
   $x=1$, predict both triples from the diagram. The fine cell
   uses its whole face 0 in both calls. The coarse cell uses
   face 1 and a different subface number in each call.
   Run the second command to check your prediction against
   `neighbor_of_coarser_neighbor()`. Within each call, compare the
   printed points from side 0 and side 1. Sum the lengths and the
   two integrals.

```bash
"$exe" --conforming
"$exe" --subfaces
```

Check that each physical piece has the following measure and contribution:

| Integration piece | Length | $\displaystyle\int_F j_F^2\,ds$ |
| --- | ---: | ---: |
| Whole interface before refinement | 1 | 16 |
| Lower half after refinement | 0.5 | 8 |
| Upper half after refinement | 0.5 | 8 |

The diagnostic already prints lengths and checks point matching.
The jump integrals become nonzero when you complete TODO 2.

## Exercise 3 — The face contributions complete the residual estimator

In the notation of [Lecture 8](../lectures/lecture-08.md), the mixed
problem adds a Neumann residual on the boundary. Let $F$ denote a
physical integration segment, with $h_F$ its length. On a
coarse–fine interface, each half is a separate $F$. With
$h_T=\operatorname{diam}T$, the indicator is

$$
e_T^2 =
h_T^2\|f+\Delta u_h\|_{0,T}^2
+\frac12\sum_{F\in\mathcal F_T^i}h_F\|j_F\|_{0,F}^2
+\sum_{F\in\mathcal F_T^N}h_F
   \|g_N-\nabla u_h\cdot n\|_{0,F}^2,
\qquad
\eta=\Big(\sum_T e_T^2\Big)^{1/2}.
$$

The sets $\mathcal F_T^i$ and $\mathcal F_T^N$ contain the interior
and Neumann segments of $\partial T$. The cell formula already in
the starter uses $\Delta(u_h|_T)=0$ for $Q_1$ on these squares.

### The mesh loop visits cells and assigns their faces to workers

`add_face_residuals()` passes the range of active cells to
`MeshWorker::mesh_loop`. A *worker* is a function that the loop calls
when it has identified the mesh entity to integrate. The flags select
the visits and the corresponding workers.

| Worker | When it runs | What it receives and computes |
| --- | --- | --- |
| Cell worker | When `assemble_own_cells` is enabled | It receives one cell and computes a cell integral. Here the third argument of `mesh_loop` is `{}` and the volume term has already been computed in `estimate()`. |
| `boundary_worker` | On a boundary face, with `assemble_boundary_faces` | It receives a cell and its local face number. On a Neumann face it integrates the squared flux mismatch, multiplies by $h_F$, and records one contribution for the adjacent cell. |
| `face_worker` | Once per interior integration segment, with `assemble_own_interior_faces_once` on our serial mesh | It receives two triples `(cell, face, subface)`, just as in Exercise 2. It integrates the squared jump and records one contribution for each adjacent cell. |

For cells at the same level, both triples specify whole faces. At a
coarse–fine interface the loop visits each segment from its fine cell.
The fine triple specifies a whole face and the coarse triple specifies
the matching subface. In the refined diagram from Exercise 2, the
lower and upper segments therefore give two separate calls to
`face_worker`.

### Scratch holds the quadrature objects and copy data holds the results

- **`scratch` → `local` in a worker.** `ScratchData<2>` is configured
  with the finite element, quadrature rules, and update flags. Each
  worker call receives the scratch object for its task. Concurrent
  tasks use separate scratch objects, and successive calls can reuse
  them. `local.reinit(cell, face)`
  returns a `FEFaceValues` for a boundary face. With two triples,
  `local.reinit(...)` returns a `FEInterfaceValues` for the common
  segment. Use the returned object while it describes that segment.
  The next `reinit` can reuse its storage for a different segment.
- **`Contributions{}` → `out` in a worker.** This is the copy data.
  It is a vector of `(active cell index, contribution to e_T squared)`
  pairs. The loop starts it empty for each active-cell visit. All
  workers called during that visit append to the same vector, so it
  can hold results from several faces and for several target cells.
  A target can be the current cell or its neighbor. Entries are
  numerical values that survive the next `reinit`.
- **`out` → `copier` → `squared_indicators`.** After the workers for
  an active-cell visit have finished, the copier reads their pairs
  and performs `squared_indicators[index] += value`. Worker calls
  can run concurrently, while copier calls are serialized. This is
  where the face terms are added to the existing volume terms and
  where contributions from different visits can reach the same cell.

For example, follow the lower segment in the refined diagram. The
face worker uses the lower fine cell's whole face 0 and the coarse
cell's face 1, subface 0, to reinitialize `local`. Once the integral
is computed, that call appends two pairs to `out`, each with half of
the weighted contribution. The copier adds one to the fine cell and
one to the coarse cell. When the upper segment is processed, its
fine cell receives its own share and the same coarse cell receives
another contribution. The factor $1/2$ divides the contribution
between the two cells because the segment is integrated once.

### Implementation and checks

1. Read the boundary worker. Point to the lines that select the
   boundary id, form the flux mismatch, append the weighted result
   to `out`, and add it to the global indicator in the copier.
2. Complete **TODO 3** in the face worker. Reuse
   `squared_normal_jump()` and compute $h_F$ as
   $\sum_q\operatorname{JxW}(q)$ on the current segment. Append
   half of $h_F\int_F j_F^2$ to `out` for each cell, using
   `active_cell_index()` for the two indices. The supplied copier
   adds these pairs to the squared cell indicators. For the refined
   two-square diagnostic, predict the additions to the coarse cell
   and to each fine cell using the table from Exercise 2.
3. Run `estimator.prm` after completing all three TODOs.
   This problem uses $u=xy(1-y)$, zero Dirichlet trace on the three
   Dirichlet edges, $f=2x$, and $g_N=y(1-y)$ on the right. Compare
   $\eta$ and the computed seminorm error as the mesh is refined.
   Open `estimator/solution.pvd` and color by the cell field
   `indicator` if you want to inspect the marked regions.

```bash
(cd estimator && "$exe" "$lab/estimator.prm")
```

With the supplied parameters, $\eta$ decreases from about $0.480$ to
$0.145$ and $|u-u_h|_1$ decreases from about $0.0840$ to $0.0250$
over four meshes. Report both quantities against the DoF count.

As an additional check, copy `neumann.prm` and set
`Use face residuals = true`. In exact arithmetic the manufactured
$Q_1$ solution has zero cell residual, zero jump, and zero Neumann
mismatch. The computed estimator should be close to zero.
