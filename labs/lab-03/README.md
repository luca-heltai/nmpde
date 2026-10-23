# Lab 3: Mesh objects, degrees of freedom, and global numbering

The complete student instructions, diagrams, API lookup sheet and exercises are
in [Laboratory 3](../../notes/laboratories/lab-03.md).

`lab-03.cc` is the **student starter**, with TODOs and no completed exercise
implementations. It uses one two-square mesh for `FE_Q(1)`, `FE_Q(2)` and
`FE_DGQ(2)`. Mesh creation and a `DataOut` helper are provided; students implement
object inspection, element inspection and DoF distribution, local-to-global
maps, renumbering and coefficient selection/transfer.

From the repository root, in the course deal.II 9.7.1 environment:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target lab-03 --parallel 2
mkdir -p build/runs/lab-03-student
cd build/runs/lab-03-student
../../bin/lab-03 | tee transcript.txt
```

The untouched starter creates `mesh.svg` and prints TODO messages, skipping
DoF-dependent work until distribution is implemented. It initially creates no
VTU files. Use the exercise checklists in the student page to verify your work.

The explanatory diagrams use illustrative numberings, independent of the
library's concrete numbering. Their SVG files are versioned in
`notes/assets/lab-03/`; regenerate them from the repository root with
`python3 labs/lab-03/make_figures.py` (standard library only).
