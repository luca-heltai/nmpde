# Laboratory 2: Meshes, connectivity, and deal.II abstractions

This is the **second deal.II laboratory**: vertices, cells, faces, connectivity,
the `Triangulation` interface, VTK export, and a qualitative preview of manifolds.
Finite element spaces, `FE_Q`, `DoFHandler`, and degrees of freedom belong to
the next laboratory.

The full guided lesson is included in the published course book:
[Laboratory 2](https://luca-heltai.github.io/nmpde/laboratories/lab-02/).

## Material

- `data/two-quads.vtk`: small hand-written legacy ASCII VTK file.
- `lab-02.cc`: runnable example using objects instead of adjacency arrays.
- `../../notes/laboratories/lab-02.md`: complete lesson, discussion and exercises.

## Build (deal.II 9.7.1 devcontainer)

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target lab-02 --parallel 2
./build/bin/lab-02 rectangle
./build/bin/lab-02 shell-flat
./build/bin/lab-02 shell-curved
```

The generated `output/*.vtk` files can be opened in ParaView. The `rectangle`
mode should report **2 cells, 6 vertices, and 6 boundary faces**. The two
quadrilaterals share one interior face, hence 7 distinct faces but 8
cell-face incidences.

The two `shell` modes use the same annular coarse mesh. The flat version
removes the automatically installed spherical manifold before refinement;
the curved version retains it. Compare the meshes in wireframe view.

The previous laboratory remains available as [Laboratory 1](../../notes/laboratories/lab-01.md).
It introduces C++, CSV output, and ParaView before deal.II is used.
