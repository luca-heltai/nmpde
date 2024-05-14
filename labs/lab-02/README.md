# Lab 2: Exploring Basis Functions in deal.II

This laboratory session is designed to help you explore the shapes of basis functions using the deal.II library. You will learn how to create and refine meshes, distribute degrees of freedom, define finite elements, and visualize the basis functions.

## Degrees of Freedom per Object

In the context of finite element methods (FEM), the term "degrees of freedom" (DoF) refers to the number of independent scalar values that are used to represent a function on a given finite element mesh. Each degree of freedom corresponds to the result of evaluating a linear functional on the mesh where the solution is approximated. The most common degrees of freedom used in FEM are function values at interpolation points (vertices, midpoint of vertices, midpoint of cells, etc.).

### Degrees of Freedom per Object

The concept of "degrees of freedom per object" refers to the distribution of degrees of freedom across various geometrical entities within the finite element mesh. These objects can include:

- **Vertices (Nodes):** Points where elements meet. In lower-order elements, degrees of freedom are often associated with the vertices.
- **Edges:** Line segments connecting vertices. Higher-order elements may have additional degrees of freedom on edges.
- **Quads/Triangles:** Surfaces bounded by edges. Higher-order elements may have degrees of freedom associated with Quads or Triangles.
- **Hexes/Tets:** Volumes bounded by Quads or Triangles. Higher-order elements can have degrees of freedom inside Hexes or Tets.

The degrees of freedom per object vector (`dpo`) is used by `DoFHandler` to decide how to distribute degrees of freedom on a `Triangulation`. It is a `dim+1` array, where the `i`-th entry indicates the number of degrees of freedom associated to objects of dimension `i`.

### Examples

1. **Linear Elements (P1 Elements):**
   - For a 2D triangular element (triangle), linear elements have degrees of freedom only at the vertices. Each vertex of the triangle represents one degree of freedom (`dpo={1,0,0}`).

2. **Quadratic Elements (P2 Elements):**
   - For a 2D triangular element with quadratic shape functions, degrees of freedom exist at the vertices and along the edges. This means each edge of the triangle has additional degrees of freedom, allowing the solution to be more accurately approximated (`dpo={1,1,0}`).

3. **Discontinuous Galerkin Elements (DG Elements):**
   - In DG methods, degrees of freedom are typically associated with the interior of the element and do not need to be continuous across element boundaries. This allows for greater flexibility in handling complex boundary conditions and discontinuities (i.e., DGP(1) on triangles: (`dpo={0,0,3}`)).

### Importance in FEM

Understanding the distribution of degrees of freedom per object is essential for:

- **Mesh Generation and Refinement:** Knowing where degrees of freedom are located helps in creating and refining meshes to better capture the solution's behavior.
- **Solution Accuracy:** More degrees of freedom generally lead to a more accurate solution, but also increase computational cost.
- **Element Selection:** Different types of elements (linear, quadratic, etc.) distribute degrees of freedom differently, affecting both the accuracy and efficiency of the solution.

## Objectives

By the end of this laboratory, you should be able to:

1. Create and refine meshes using the `Triangulation` class.
2. Generate different types of grids using the `GridGenerator` namespace.
3. Set up and distribute degrees of freedom using the `DoFHandler` class.
4. Define finite elements with the `FiniteElement` class, specifically `FE_Q` and `FE_DGQ`.
5. Output and visualize the shapes of basis functions using the `DataOut` class.

## Prerequisites

Make sure you have the following before you start:

- A working installation of the deal.II library.
- Basic knowledge of C++.

## Example Code

Here is a simple example to get you started. This code creates a subdivided hyper-rectangle mesh, distributes degrees of freedom, and outputs the shapes of basis functions to a VTU file.

```cpp
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <iostream>

using namespace dealii;

int main()
{
  // Create a 2D triangulation object
  Triangulation<2> triangulation;

  // Define a finite element space with polynomial degree 2
  FE_Q<2> fe(2); // dpo = {1, 1, 1}
  // FE_DGQ<2> fe(2); // dpo = {0, 0, 9}

  // Set up the DoF handler with the triangulation
  DoFHandler<2> dof_handler(triangulation);

  // Generate a subdivided hyper-rectangle mesh
  GridGenerator::subdivided_hyper_rectangle(triangulation,
                                            {2, 1},
                                            Point<2>(0, 0),
                                            Point<2>(2, 1));

  // Distribute degrees of freedom among the cells
  dof_handler.distribute_dofs(fe);

  // Output the number of active cells and total cells in the triangulation
  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "Total number of cells: " << triangulation.n_cells()
            << std::endl;

  // Output the number of degrees of freedom
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  // Output the number of dofs per cell
  std::cout << "Number of dofs per cell: " << fe.dofs_per_cell << std::endl;

  // Output the global dof indices for each cell
  std::vector<types::global_dof_index> dofs_per_cell(fe.dofs_per_cell);
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      std::cout << cell << std::endl;
      cell->get_dof_indices(dofs_per_cell);
      for (const auto &dof : dofs_per_cell)
        {
          std::cout << dof << " ";
        }
      std::cout << std::endl;
    }

  // Create a DataOut object to visualize the basis functions
  DataOut<2> data_out;

  // Set flags to write higher order cells
  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;
  data_out.set_flags(flags);

  // Attach the DoF handler to the DataOut object
  data_out.attach_dof_handler(dof_handler);

  // Create and add basis functions to the DataOut object
  std::vector<Vector<double>> basis_functions(
    10, Vector<double>(dof_handler.n_dofs()));
  for (unsigned int i = 0; i < 10; ++i)
    {
      Vector<double> &basis_function = basis_functions[i];
      basis_function[i]              = 1;
      data_out.add_data_vector(basis_function,
                               "basis_function_" + std::to_string(i));
    }

  // Build patches and write the output to a VTU file
  data_out.build_patches(fe.degree);
  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);
}
```

## Exercises

### Exercise 1: Visualizing Basis Functions for Different Degrees

1. **Basis Functions for Degree 1:**
   - Modify the example code to use `FE_Q<2>(1)` and visualize the first few basis functions.
   - Observe and describe the shapes of the basis functions in the generated VTU files.

2. **Basis Functions for Higher Degrees:**
   - Change the finite element to `FE_Q<2>(3)` and visualize the first few basis functions.
   - Compare the shapes of the basis functions for polynomial degrees 1, 2, and 3.

### Exercise 2: Exploring Discontinuous Galerkin Basis Functions

1. **Visualizing DG Basis Functions:**
   - Replace `FE_Q` with `FE_DGQ` in the example code and visualize the basis functions.
   - Observe how the shapes of the DG basis functions differ from the continuous basis functions.

2. **Comparing Continuous and Discontinuous Basis Functions:**
   - Visualize and compare the basis functions for `FE_Q<2>(2)` and `FE_DGQ<2>(2)`.
   - Describe the differences in their shapes and explain why they differ.

### Exercise 3: Basis Functions on Different Meshes

1. **Hyper Shell Mesh:**
   - Modify the example code to create a hyper shell mesh using `GridGenerator::hyper_shell`.
   - Visualize the basis functions on the hyper shell mesh and describe any differences compared to the hyper-rectangle mesh.

2. **Adaptive Mesh Refinement:**
   - Implement a strategy to refine the mesh adaptively around a specific region (e.g., near the origin).
   - Visualize the basis functions on the adaptively refined mesh and compare them to those on a uniformly refined mesh.

### Exercise 4: Higher-Order Elements and Basis Functions

1. **Using `FE_Nedelec` Elements:**
   - Replace the finite element in the example code with `FE_Nedelec` and visualize the basis functions.
   - Describe the shapes of the Nedelec basis functions and their applications.

2. **Using `FE_RaviartThomas` Elements:**
   - Replace the finite element with `FE_RaviartThomas` and visualize the basis functions.
   - Compare the shapes of the Raviart-Thomas basis functions with those of `FE_Q` and `FE_Nedelec`.

### Exercise 5: Output and Visualization

1. **Output Grids in Different Formats:**
   - Output the grid and basis functions to different formats such as `SVG`, `EPS`, and `VTK` using `GridOut` and `DataOut`.
   - View the generated files using appropriate viewers and compare the visualizations.

2. **Visualizing Basis Functions in Paraview:**
   - Open the VTU files in Paraview and explore the different visualization options.
   - Create contour plots, vector field plots, and other visualizations of the basis functions.
