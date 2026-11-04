# Reproduce the harmonic corner experiment

This experiment accompanies Lecture 6. It solves `-Delta u = 0` with continuous
piecewise affine elements on five uniformly refined triangular meshes. In polar
coordinates, with `theta0 = 2*pi - omega`, the exact solution is

```text
u(r,theta) = r^alpha sin(alpha (theta-theta0)),  alpha = pi/omega.
```

The apertures are `2*pi`, `7*pi/4`, `3*pi/2`, `5*pi/4`, and `pi`. Dirichlet
values are the nodal interpolant of this function on **every** boundary edge.
The rays carry zero data; the outer polygon carries nonzero data. The right-hand
side is zero. The `omega=pi` case has the affine solution `u=-y`: its errors
measure roundoff and solver tolerance and are excluded from slope fits.

## Build and compute

Run from the repository root in the course deal.II 9.7.1 environment (for example,
the existing devcontainer). This uses the existing GTest dependency and native
serial deal.II linear algebra, with CG and SSOR; FSI-suite and Gmsh are unnecessary.

```bash
cmake -S . -B build/pacman-release -DCMAKE_BUILD_TYPE=Release \
  -DNMPDE_BUILD_LABS=OFF -DNMPDE_BUILD_DEALII_LABS=OFF \
  -DNMPDE_BUILD_TESTS=ON
cmake --build build/pacman-release --target nmpde_pacman_tests --parallel 2
ctest --test-dir build/pacman-release -L pacman --output-on-failure
```

Or run `make pacman`. Existing complete case outputs are reused and the
corresponding GTest cases are marked skipped. For a fresh computation and
numerical validation, use `make pacman REGENERATE_FIGURES=1` or set
`NMPDE_REGENERATE_FIGURES=1` for the direct executable/CTest invocation. Only the new target is compiled. To compute just the
`3*pi/2` case:

```bash
./build/pacman-release/bin/nmpde_pacman_tests \
  --gtest_filter='*ConvergenceAndVtu/2'
```

Each computed case writes its own directory under
`notes/assets/generated/lecture-06/pacman-N/`:

- `convergence.csv`: six levels, cells, DoFs, maximum cell diameter, L2 error,
  H1 seminorm error, full H1 error, successive rates, residual, quadrature checks;
- `solution-0.vtu` through `solution-5.vtu`, and `solution.pvd` for ParaView;
- `metadata.txt`: deal.II version, discretization, quadrature and solver settings.

When a case is computed, the test checks the affine control, decreasing errors, broad asymptotic rate
windows, relative algebraic residual below `2e-11`, and relative changes below
`5e-4` when the finest-mesh error quadrature increases from order 8 to 12.
The CSV's quadrature columns are zero on levels where the check is not run;
only the final row of the four singular cases is a measured comparison.
Failures are reported by GTest. Read the test output before using the figures.

## Generate convergence plots and tables

Python with NumPy and Matplotlib is sufficient:

```bash
python3 -m pip install numpy matplotlib
python3 tests/pacman/plot_convergence.py
```

Or run `make pacman-plots PYTHON=python3` after computing the data. This produces
`convergence-h1.svg`, `convergence-l2.svg`, `tables.md` and `fitted-rates.csv` in
`notes/assets/generated/lecture-06/`. The figures show errors against both `h`
and the number of DoFs `N`. Dashed lines indicate reference powers; they are
anchored at the final data point, not fitted. Fitted exponents in the table use
the three finest levels. The rates in each CSV use consecutive values of `h`.
The H1 figure uses the **seminorm**; the full norm is also retained in the CSV.
No historical numerical table is used as input.

## Reproduce the ParaView images

Open a `solution.pvd` file in ParaView and choose a refinement level. The PVD
"time" is the cycle index, not physical time. The arrays are:

- `solution`: computed finite element solution;
- `exact_nodal_interpolant`: the nodal interpolant of the analytic solution,
  **not** the analytic function between nodes;
- `nodal_error`: their signed difference at the nodes;
- `cell_H1_error`: the integrated H1 seminorm error on each cell, neither a
  pointwise gradient error nor an a posteriori estimator.

The deterministic rendering script generates missing images among the two five-panel images used
by the lecture, using cycle 1 (one global refinement) and common colour scales:

```bash
pvpython --disable-registry tests/pacman/render_paraview.py
# macOS with the installation used for the current images:
/Applications/ParaView-6.1.0.app/Contents/bin/pvpython --disable-registry \
  tests/pacman/render_paraview.py
```

Or run `make pacman-images PVPYTHON=/path/to/pvpython`. Use `--cycle 2 --force` to display
a finer mesh and `--output output/pacman-images` to preserve the lecture images.
On a headless Linux installation, ParaView needs a functioning offscreen backend;
if that build uses X11, invoke it with `xvfb-run -a pvpython ...`. Rendering is
optional for the numerical tests. The committed PNG snapshots can be recreated with `--force`
from the local VTU files; no external ParaView state or absolute data path is used.

## Mesh provenance and geometry

The five unmodified mesh fixtures (31 KB total) come from Luca Heltai's
[FSI-suite](https://github.com/luca-heltai/fsi-suite), commit
`4788d574d6d8c24aa410f4ea2a166cc0b24bfe53`, `grids/pacman_0.msh` through
`grids/pacman_4.msh`. The original experiments are
`fsi_book/chapter_2.cc`, `Chapter2.Example2_1_Pacman_0` through `_4`, with inputs
under `book_prms/chapter_2_example_2_1_pacman_*.prm`. The original mesh-generation
recipe is `grids/pacman.geo`. The five retained meshes fully specify the domains
and their topology. Their SHA-256 checksums are in `meshes/SHA256SUMS`.

The present solver and plotting/rendering scripts are independent course code;
FSI-suite's application framework and parameter infrastructure are not copied.
The initial outer circular boundary is represented by chords and all manifold
ids are set to flat. Global refinements therefore solve on the **same polygon**
at every level, with no moving-boundary geometry error. Since the exact function
is harmonic on each polygon, it remains an exact solution there with its own
Dirichlet trace. Numerical values need not coincide with the book's historical
outputs, which used different solver and error-quadrature settings.

The first mesh represents the limiting slit case, with opposite banks separated
by about `1e-15` in the original coordinates. Preserve its distinct vertices;
do not merge them or replace this mesh by an uncut disk. This case is a numerical
limit example, outside the bounded Lipschitz-domain hypotheses of the lecture's
main regularity statements.

For `omega>pi`, the reference exponents are `alpha` in the energy seminorm and
`2*alpha` in L2, with the strict subcritical regularity qualification explained
in the lecture. With quasi-uniform 2D refinement, `N` grows like `h^-2`, so the
corresponding log-log slopes against DoFs are `-alpha/2` and `-alpha`. Nodal
Dirichlet approximation is included in the computed error. The homogeneous-data
Aubin-Nitsche theorem alone does not account for that boundary term; the lecture
states the required additional qualification. These tests provide numerical
evidence and do not establish endpoint regularity or an infimum over all meshes.

## Incremental generation and the CI keyword

All generators preserve existing nonempty outputs. A missing output triggers
only the required case or figure; existing files are not overwritten. Plotting
and rendering scripts accept `--force` to replace all their outputs. For example:

```bash
python3 tests/pacman/plot_convergence.py --force
pvpython --disable-registry tests/pacman/render_paraview.py --force
```

In CI, `[regenerate figures]` in a push commit message, a PR title/body, or a
commit in the PR range forces a new numerical validation and regenerates all
figures. The merged PR title/body is checked on the main-branch push too.
Without it, persistent cached outputs are reused and missing/empty
files are generated. A cold or expired cache can therefore require generation.
Use the keyword after changes to the solver, meshes or figure parameters.
