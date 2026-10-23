# Laboratory 3 — Mesh objects, degrees of freedom, and global numbering

This laboratory connects the local finite element triple from
[Lecture 4](../lectures/lecture-04.md) to a global discrete function in deal.II.
You will keep one mesh fixed while changing the finite element and the DoF
numbering. The deliverables are your completed source, prediction sheet,
index tables, checks and a small set of VTU files. The starter contains TODOs;
this page does not provide completed exercise implementations or answer tables.

## 1. Objectives and working assumptions

You should be able to:

- distinguish mesh-object indices, local basis indices and global DoF indices;
- read an element's degrees of freedom per object (`dpo`) and predict counts;
- explain what `DoFHandler::distribute_dofs()` adds to a `Triangulation`;
- reconstruct a function from global coefficients and local shape functions;
- renumber DoFs while preserving topology and the represented function.

We use deal.II **9.7.1**, C++17, a serial mesh, a fixed scalar element on all
cells, no refinement, no periodic identifications and no boundary constraints.
All DoFs remain numbered, including boundary DoFs. Higher dimensions are an
optional conceptual extension. Quadrature, assembly and hanging-node
constraints belong to later laboratories.

## 2. Three index systems describe different objects

A mesh stores geometric and topological objects: vertices, edges, faces and
cells. Their indices describe connectivity, independently of which finite
element you later choose. A finite element has a *local* ordered basis and
dual basis. Distributing DoFs then assigns *global* indices to the independent
coordinates of the assembled finite element space.

```{figure} ../assets/lab-03/mesh-entities.svg
:alt: Local vertex and face slots on a reference square compared with global mesh-object labels on a separate three-cell mesh; no DoF numbers are assigned.
:name: fig:lab-03-objects

Local subobject indices restart on each cell. Mesh-object indices identify
the shared vertices and edges. The displayed global labels are illustrative;
they belong to a three-cell example, separate from the exercise mesh.
```

```{figure} ../assets/lab-03/numbering-layers.svg
:alt: A one-dimensional example showing mesh vertex indices, local Q2 basis indices, and a different global DoF numbering.
:name: fig:lab-03-numbering

A worked one-dimensional illustration. The arbitrary global numbering differs
from the mesh numbering. Both cells refer to the shared global DoF. This is a
schematic example, not a transcript of deal.II's default numbering. The exercises
below use a different, two-dimensional mesh.
```

With the convention of Lecture 4, write

$$
v_h=\sum_{I=0}^{N_h-1}U^I\Phi_I,\qquad
v_h|_K=\sum_{i=0}^{n_K-1}U^{g_K(i)}\varphi_{K,i}.
$$

Here $I$ is a global DoF index, $i$ is a local basis index, and
$g_K(i)$ maps the latter to the former. The coefficient $U^I$ is stored in a
`Vector<double>` entry, `coefficients[I]`. In this lab the global dual
functionals use point evaluations, $U^I=\ell_h^I(v_h)$. For a discontinuous
element, an evaluation also specifies its owning cell: at an interface the
two cellwise polynomial traces can differ. A physical point alone therefore
does not specify a DG functional. DoFs can also be moments, as in Lecture 4's
optional Raviart–Thomas extension.

A cell-local vertex number and a cell-local basis number are different types
of index, even when some small numerical values happen to coincide.
Likewise, an object index does not encode a coordinate or a DoF number.
Compare quantities explicitly, without relying on apparent numbering patterns.

## 3. Degrees of freedom per object are exclusive counts

For an element in dimension $d$, `dpo[r]` counts the DoFs **logically attached
to one object of dimension $r$**, excluding its lower-dimensional boundary
objects. For the uniformly chosen elements in this lab, every object of the
same dimension has the same count.

| Object dimension | 2D quadrilateral mesh | 3D hexahedral mesh |
|---|---|---|
| 0 | Vertex | Vertex |
| 1 | Edge, also called a face in 2D | Edge |
| 2 | Quadrilateral cell | Quadrilateral surface face |
| 3 | — | Hexahedral volume cell |

Thus “face” means a codimension-one object: a line segment in 2D, a surface
patch in 3D. Do not count a 2D cell again as one of its faces.

```{figure} ../assets/lab-03/dpo-objects.svg
:alt: A Q2 square whose vertex, edge-interior, and cell-interior DoFs have different colors, explaining exclusive versus inclusive counts.
:name: fig:lab-03-dpo

The exclusive edge count excludes endpoint DoFs. The inclusive face count
includes them. The inclusive cell count includes the DoFs on all its subobjects.
```

For a conforming mesh under the assumptions of §1, if $N_r$ counts the
**distinct** mesh objects of dimension $r$, then

$$
N_h=\sum_{r=0}^d N_r\,\mathrm{dpo}[r].
$$

For one quadrilateral, count its vertices, edges and cell interior separately
to obtain the analogous local count. Counting incidences by looping over all
cell faces counts an interior face twice; it does not give $N_1$.

For tensor-product Lagrange elements, derive the exclusive counts by
classifying the tensor grid according to which coordinates are endpoints
and which are interior. For discontinuous elements, inspect their declared
attachment to objects. A support point's physical position does not determine
its logical attachment.

```{figure} ../assets/lab-03/sharing.svg
:alt: Shared face functionals in a conforming space compared with independent cell-local functionals in a discontinuous space, even at coincident support points.
:name: fig:lab-03-sharing

Conforming face functionals are identified across the common face.
Discontinuous cell-local functionals remain independent. The DG markers are
slightly offset only to make both visible; their support points can coincide.
```

An element's `dpo` describes its attachment pattern. The `DoFHandler` combines
that pattern with mesh connectivity to construct global numbering. For the
scalar elements studied here, shared nodal measurements enforce the matching
trace property proved in Lecture 4. A count by itself is not a proof of
conformity for an arbitrary finite element family.

## 4. The roles of the three deal.II classes

```{figure} ../assets/lab-03/handler.svg
:alt: Triangulation provides connectivity, FiniteElement provides the local element and dpo, DoFHandler provides global numbering, and a separate vector stores coefficients.
:name: fig:lab-03-handler

`DoFHandler` organizes the global coordinates of a finite element space over an
existing mesh. A coefficient vector stores the values defining one function.
```

| Object | Information it supplies in this lab |
|---|---|
| `Triangulation<2>` | Vertices, coordinates, connectivity, object indices and active cells |
| `FiniteElement<2>` (`FE_Q` or `FE_DGQ`) | Local shape functions, local DoF definitions and attachment counts |
| `DoFHandler<2>` | A finite element assignment to the mesh, global DoF indices, and maps $g_K$ |
| `Vector<double>` | The coefficients $U^I$ of one discrete function |

Creating a `DoFHandler` attached to a mesh does not yet distribute DoFs.
`distribute_dofs(fe)` performs that operation using the supplied finite element.
The handler's cell iterators expose both mesh information and DoF information.
A `Triangulation` cell iterator alone does not provide a DoF numbering.

Each of the starter's three cases constructs a fresh handler over the **same**
triangulation. No mesh is regenerated between cases. A change of element can
change the dimension or continuity of the space. Renumbering an already
constructed handler changes the coordinate ordering within the same space.

## 5. Build and run the student starter

The source is `labs/lab-03/lab-03.cc`. Use the course devcontainer based on
`dealii/dealii:v9.7.1-noble`, or the configured deal.II environment used for the
previous laboratory. Run the following from the **repository root**:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target lab-03 --parallel 2
mkdir -p build/runs/lab-03-student
cd build/runs/lab-03-student
../../bin/lab-03 | tee transcript.txt
```

The last two commands run the executable from a dedicated output directory.
After editing, return to the repository root, rebuild `lab-03`, and run it
again from that output directory. Use a different run directory to retain an
earlier transcript or VTU files.

The unchanged starter must compile and finish successfully. It creates
`mesh.svg`, prints headings for `q1`, `q2` and `dgq2`, and prints explicit
`[TODO ...]` messages. It skips DoF-dependent work while TODO 1c is unfinished.
It does **not** initially create any VTU files. Successful compilation at this
stage checks the scaffold; it does not complete the exercises.

The cases use `FE_Q<2>(1)`, `FE_Q<2>(2)` and `FE_DGQ<2>(2)` on two adjacent
unit squares forming $[0,2]\times[0,1]$. Do not refine or modify this mesh in
the essential exercises.

### API lookup sheet for deal.II 9.7.1

The following are API names and contracts to consult, not completed exercise code.

| Task | API and interpretation |
|---|---|
| Traverse active cells | `tria.active_cell_iterators()` or `dof_handler.active_cell_iterators()` |
| Identify a cell | `cell->level()`, `cell->index()`; the pair identifies a cell in a triangulation hierarchy |
| Traverse local vertices/faces | `cell->vertex_indices()`, `cell->face_indices()` |
| Mesh vertex and its coordinates | `cell->vertex_index(v)`, `cell->vertex(v)` |
| Mesh face and endpoint vertex IDs | `cell->face(f)->index()`, `cell->face(f)->vertex_index(v)` |
| Exclusive counts | `fe.n_dofs_per_vertex()`, `fe.n_dofs_per_line()`, `fe.n_dofs_per_quad()` |
| Inclusive counts | `fe.n_dofs_per_face()`, `fe.n_dofs_per_cell()` |
| Distribute and count global DoFs | `dof_handler.distribute_dofs(fe)`, `dof_handler.n_dofs()` |
| Obtain $g_K$ | `cell->get_dof_indices(indices)`; resize `indices` to the local DoF count first |
| DoF attached to a vertex | `cell->vertex_dof_index(v, j)`; $j$ selects one DoF attached to that vertex |
| Apply a permutation | `dof_handler.renumber_dofs(new_numbers)`; entry at an old index gives its new index |

Use `types::global_dof_index` for global DoF indices. Do not call
`vertex_dof_index()` on `FE_DGQ`: it has no DoFs attached to vertices.
For the scalar `FE_Q` elements in this lab, use `j = 0` to select the single
DoF attached to a vertex. To list the endpoints of a face, traverse that
face's own `vertex_indices()` and call its `vertex_index()`.
`get_dof_indices()` returns global indices in the element's local basis order;
it does not sort them in increasing global order. The particular global
numbers can vary with traversal and renumbering.

## 6. Exercise 1 — Separate topology from element attachment

Complete TODOs **1a, 1b and 1c**, in that order.

1. Before changing the source, draw the two-square mesh. Assign your own
   vertex and edge labels and list the vertex and edge incidences of each
   cell. Mark the shared edge and its endpoints.
2. In `inspect_topology()`, print one row per local vertex with columns
   `(cell level, cell index, local vertex, mesh vertex index, x, y)`.
   Print one row per local face with columns
   `(cell level, cell index, local face, mesh face index, endpoint IDs)`.
3. Insert the reported vertex and face IDs into separate `std::set` objects.
   Record their sizes. Also record the number of face incidences and explain
   why it differs from the number of distinct faces.
4. Fill the following prediction sheet on paper. The first three numerical
   columns contain exclusive attachment counts. Derive them from the finite
   element definitions before querying the API.

| Element | Vertex | Edge interior | Cell interior | DoFs per face | DoFs per cell | Global DoFs |
|---|---|---|---|---|---|---|
| `FE_Q<2>(1)` | … | … | … | … | … | … |
| `FE_Q<2>(2)` | … | … | … | … | … | … |
| `FE_DGQ<2>(2)` | … | … | … | … | … | … |

5. Implement `inspect_element()` using the lookup sheet. Then implement the
   single missing distribution operation in `run_case()`. Rebuild and run.
6. Compare the reported values with your predictions. Check that the local
   inclusive count equals the sum of the exclusive contributions of all cell
   subobjects, and that the global count agrees with §3's formula using
   **distinct** objects. Explain each discrepancy before proceeding.

**Deliverable:** the completed prediction sheet, topology tables and a short
explanation of exclusive versus inclusive counts. The topology must be the
same for all three finite elements. Do not impose boundary conditions.

## 7. Exercise 2 — Inspect local-to-global maps

Complete TODO **2** in `inspect_dofs()`.

1. Allocate an index vector with the number of local DoFs of the handler's
   element. For each active cell, fill it using `get_dof_indices()`.
2. Print rows `(cell level, cell index, local basis index i, global DoF index I)`.
   Keep the local order; do not replace it by a sorted list.
3. Create a set from each cell's global indices. Compute and report the
   intersection of the two sets, using either `std::set_intersection()` or
   a loop with membership tests. Record its cardinality for each element.
4. For each `FE_Q` case, compare the mesh vertex index and the global DoF
   index attached to that vertex. Find the two occurrences of each shared
   vertex. Explain what is shared and which numbers have no reason to agree.
5. Relate the shared DoFs to trace matching. For `FE_DGQ`, explain why
   coincident evaluation points do not imply shared global coordinates.
   Do not attempt to identify DG DoFs by calling `vertex_dof_index()`.

**Checks:** every reported global index is in `[0, n_dofs())`; a cell's list
has the expected length and no duplicate global indices in this experiment;
the union of both cell sets covers all global DoFs. The sum of the two list
lengths minus the intersection size must agree with the global count.
These relations provide checks without prescribing a global numbering.

**Deliverable:** the three pairs of local-to-global tables and their shared-ID
sets. Explain the difference between a global basis function and its local
restrictions using the notation $\Phi_I$, $\varphi_{K,i}$ and $g_K$.

## 8. Exercise 3 — Renumber coordinates and preserve a function

Complete TODOs **3a and 3b** in `renumber_and_visualize()`.

1. Choose one global basis function per element. For `FE_Q`, choose a DoF
   attached to the shared edge or one of its endpoints; use Exercise 2's
   tables to identify it. For `FE_DGQ`, choose one cell-local DoF and record
   which cell contains it. State the selected global index in your transcript.
2. Construct the coefficient vector that represents that single global basis
   function. Its size must be `dof_handler.n_dofs()`. Use the supplied
   `write_field()` helper to write `<tag>-before.vtu`, where `<tag>` is
   `q1`, `q2` or `dgq2`. A fixed-size loop over ten basis functions is not used.
3. Save both local-to-global tables and the corresponding **local coefficient
   lists**, one entry per local basis function. Also retain the topology table.
4. Build a reverse permutation of all global DoF indices: the first index
   becomes the last, the second becomes the penultimate, and so on. Store the
   permutation in a `std::vector<types::global_dof_index>` of the required size.
   Before applying it, verify its entries are in range and all distinct.
5. Read the `renumber_dofs()` contract in the lookup sheet and apply the
   permutation. Print the new local-to-global tables. Check the mesh-object
   indices, connectivity, number of DoFs, and shared-set cardinalities remain
   unchanged. The shared global ID values may change.
6. Determine how to transfer the old coefficient entries to the new indices.
   `renumber_dofs()` does not modify your `Vector<double>`. Construct a separate
   new vector, then use `write_field()` to create `<tag>-after.vtu`.
7. Using the new maps, extract the new local coefficient lists. Compare them
   with the saved lists in the same local basis order. For this permutation
   and these coefficients, require exact equality. Explain why local
   unisolvence makes this a quantitative check of function preservation.
8. In ParaView, open both VTU files, click **Apply**, select `field` and use
   **Surface With Edges**. Compare before/after with identical color ranges.
   A `Warp By Scalar` filter can make the basis shape easier to inspect.
   Record which cells support the selected basis; compare the support with
   your index tables. Visual agreement complements the coefficient check.

**Deliverable:** permutation checks, before/after index tables, matching local
coefficient lists and six VTU files. Explain why equal vector entries at equal
array positions need not represent the same function after a renumbering.

## 9. Optional extensions

- Extend the object-count table to a **single 3D hexahedron** with `FE_Q<3>(2)`.
  Distinguish exclusive edge, surface-face and volume-interior counts from the
  inclusive counts. Predict the corresponding entries for `FE_Q<3>(3)`.
- Replace reversal by a different bijection. Verify the same invariants.
- Export two basis functions whose associated objects differ and compare their
  supports. Retain the local-coefficient checks when renumbering them.

Do not add adaptive refinement or boundary constraints in these exercises:
those change the assumptions behind the simple object-count formula.

## 10. Tutorial and video references

The links below identify the relevant sections; implementing a PDE solver or
sparsity pattern is not required for this laboratory.

- **deal.II step-1:** mesh creation and cell/vertex traversal, reviewed in
  [Laboratory 2](lab-02.md). Consult the
  [commented source for 9.7.1](https://github.com/dealii/dealii/blob/v9.7.1/examples/step-1/step-1.cc).
- **deal.II step-2, main reference:** read “Enumerating degrees of freedom” in
  the [tutorial introduction](https://github.com/dealii/dealii/blob/v9.7.1/examples/step-2/doc/intro.dox).
  In the [commented program](https://github.com/dealii/dealii/blob/v9.7.1/examples/step-2/step-2.cc),
  focus on “Creation of a DoFHandler” and “Renumbering of DoFs”. The tutorial's
  sparsity sections explain a later application of the numbering.
- **deal.II step-3, output reference:** locate `output_results()` in the
  [9.7.1 commented program](https://github.com/dealii/dealii/blob/v9.7.1/examples/step-3/step-3.cc)
  to see the `DataOut` pattern used by our supplied output helper. The solver
  and assembly sections are for a later laboratory.
- **Wolfgang Bangerth, video lecture 4:**
  [The building blocks of a finite element code](https://www.math.colostate.edu/~bangerth/videos.676.4.html),
  for the relationship between mathematical concepts and software classes.
- **Wolfgang Bangerth, video lecture 9:**
  [step-2: Degrees of freedom, sparsity in finite element matrices](https://www.math.colostate.edu/~bangerth/videos.676.9.html),
  the principal video companion for enumeration and its role in finite element
  matrices. Focus on DoF enumeration; sparsity is background here.
- **Video lecture 32 in Wolfgang's series, presented by Timo Heister:**
  [ParaView, an alternative visualization tool](https://www.math.colostate.edu/~bangerth/videos.676.32.html),
  for opening data sets and inspecting scalar fields.

The video pages also link to slides. They describe older versions of the
software; use the version-pinned 9.7.1 source and API contracts for code details.

## 11. Student starter source

The source below is included directly from the compilable starter. Complete
its TODOs in your working copy; no reference solution is included.

```{literalinclude} ../../labs/lab-03/lab-03.cc
:language: cpp
```
