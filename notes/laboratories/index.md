# Laboratories

Laboratory material is released progressively during the course. The
corresponding source code is kept in the repository [`labs/` directory](https://github.com/luca-heltai/nmpde/tree/main/labs)
and can be built in the provided deal.II devcontainer.

Start with [Laboratory 1 — From a C++ Source File to a Visualized Data Set](lab-01.md),
which introduces C++, CSV output, and ParaView. [Laboratory 2 — Meshes,
connectivity, and deal.II abstractions](lab-02.md) then introduces VTK,
`Triangulation`, cell iterators, and manifolds.
[Laboratory 3 — Mesh objects, degrees of freedom, and global numbering](lab-03.md)
distinguishes mesh-object indices from local and global DoF indices, develops
degrees of freedom per object, and explores renumbering with a student starter.
[Laboratory 4 — From cell integrals to a Poisson solver](lab-04.md)
uses mapping and quadrature to assemble local contributions, impose Dirichlet
data and complete a solver following deal.II step-3.
[Laboratory 5 — Interpolation errors and convergence](lab-05.md)
reconstructs an interpolant at quadrature points, computes its error in
$L^2$ manually, and studies convergence for several polynomial degrees.
The final step introduces deal.II error integration and the full $H^1$ norm.
