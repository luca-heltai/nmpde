# Lab 4: From cell integrals to a Poisson solver

The student instructions, formulas, API lookup sheet and three exercises are
in [Laboratory 4](../../notes/laboratories/lab-04.md). They build directly on
[Lecture 5](../../notes/lectures/lecture-05.md).

`step-3-starter.cc` is the **student starter**, with TODOs for integration,
sparsity, local assembly, local-to-global accumulation, Dirichlet data and the
CG call. The mesh, class structure, diagnostics and output helper are supplied.
There is no completed exercise implementation in the student material.

From the repository root, in the course deal.II **9.7.1** environment:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target lab-04 --parallel 2
mkdir -p build/runs/lab-04-student
cd build/runs/lab-04-student
../../bin/lab-04 1 | tee transcript-r1.txt
```

The optional argument is the number of global refinements, from 1 to 5;
the default is 3. The untouched starter writes `mesh.svg`, reports an
unfinished area integral and stops before allocating or solving the system.
After the TODOs are complete, it also writes `solution.vtu`. Run each mesh in
a separate directory to retain its output. The exercise checklists give the
expected counts, invariants and tolerances.

The root target `lab-04` now builds `step-3-starter.cc`. The historical
interpolation program remains available, unchanged, as `lab-04.cc`;
it is separate from this laboratory's exercises.
