# Introduction to deal.II: A Hands-On Laboratory

This laboratory session is designed to introduce you to some of the fundamental components of the deal.II library: `Triangulation`, `DoFHandler`, `FiniteElement`, `GridGenerator`, and `GridOut`. You will explore how to create and manipulate meshes, distribute degrees of freedom, define finite elements, and visualize grids.

## Objectives

By the end of this laboratory, you should be able to:

1. Create and refine meshes using the `Triangulation` class.
2. Generate different types of grids using the `GridGenerator` namespace.
3. Set up and distribute degrees of freedom using the `DoFHandler` class.
4. Define finite elements with the `FiniteElement` class.
5. Output and visualize grids using the `GridOut` class.

## Prerequisites

Make sure you have the following before you start:

- A working installation of the deal.II library.
- Basic knowledge of C++.

## Example Code

Here is a simple example to get you started. This code creates a hyper shell mesh, distributes degrees of freedom, and outputs the grid to an SVG file.

```cpp
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>
#include <iostream>

using namespace dealii;

int main()
{
  // Define the center of the circle
  Point<2> center;
  double inner_radius = 1.0;
  double outer_radius = 2.0;

  // Create a 2D triangulation object
  Triangulation<2> triangulation;
  FE_Q<2> fe(2);
  DoFHandler<2> dof_handler(triangulation);

  // Generate a hyper shell mesh (annulus) with specified radii
  GridGenerator::hyper_shell(triangulation, center, inner_radius, outer_radius);

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

  // Create a GridOut object to write the grid to a file
  GridOut grid_out;
  std::string filename = "grid-1.svg";
  std::ofstream out(filename);
  grid_out.write_svg(triangulation, out);
}
```

## Exercises

### Exercise 1: Creating and Refining Meshes

1. **Creating a Simple Mesh:**
   - Modify the example code to create a simple 2D square mesh using `GridGenerator::hyper_cube`.
   - Print the number of active cells and total cells.

2. **Refining the Mesh:**
   - Refine the mesh globally using `triangulation.refine_global()`.
   - Print the number of active cells and total cells after each refinement step.
   - Experiment with different levels of global refinement.

### Exercise 2: Generating Different Types of Grids

1. **Hyper Shell Mesh:**
   - Create a hyper shell mesh as shown in the example.
   - Refine the mesh globally and print the number of active and total cells.

2. **Other Grid Types:**
   - Use `GridGenerator::subdivided_hyper_rectangle` to create a subdivided hyper-rectangle mesh.
   - Use `GridGenerator::hyper_ball` to create a hyper ball mesh.
   - Print the number of active cells and total cells for each type of mesh.

### Exercise 3: Setting Up Degrees of Freedom

1. **Simple DoF Distribution:**
   - Use the `FE_Q` finite element with a polynomial degree of 1.
   - Distribute the degrees of freedom using the `DoFHandler`.
   - Print the number of degrees of freedom.

2. **Higher-Order Finite Elements:**
   - Change the polynomial degree of the `FE_Q` finite element to 2 and then to 3.
   - Print the number of degrees of freedom for each case.
   - Observe how the number of degrees of freedom changes with the polynomial degree.

### Exercise 4: Exploring Different Finite Elements

1. **Using Different Finite Elements:**
   - Use `FE_Q` and `FE_DGQ` finite elements with various polynomial degrees.
   - Compare the number of degrees of freedom and the number of active cells.

2. **Custom Finite Elements:**
   - Experiment with other finite element classes available in deal.II, such as `FE_Nedelec` and `FE_RaviartThomas`.
   - Understand the differences in their usage and applications.

### Exercise 5: Output and Visualization

1. **Output Grids in Different Formats:**
   - Output the grid to different formats such as `SVG`, `EPS`, and `VTK` using `GridOut`.
   - View the generated files using appropriate viewers.

2. **Visualizing Refined Meshes:**
   - Refine the mesh locally based on some criteria (e.g., distance from the origin).
   - Output and visualize the refined mesh.
   - Observe how the refinement affects the mesh structure.

### Exercise 6: Advanced Mesh Manipulations

1. **Local Refinement:**
   - Implement a local refinement strategy where only cells near the center of the mesh are refined.
   - Print the number of active cells and total cells before and after refinement.

2. **Mesh Smoothing:**
   - Explore mesh smoothing techniques to improve the quality of the mesh.
   - Apply smoothing to a refined mesh and observe the changes.

## Conclusion

By completing these exercises, you will gain a deeper understanding of the key classes and functions in the deal.II library. This hands-on experience will prepare you for more advanced topics and applications in numerical simulations using deal.II.

Feel free to explore the deal.II documentation and tutorials for more detailed explanations and advanced examples.
