# Lab: Interpolating Functions and Computing Errors in deal.II

This laboratory session is designed to help you get familiar with some fundamental operations in the deal.II library: interpolating functions and computing errors. You will explore these concepts through a series of exercises.

## Objectives

By the end of this laboratory, you should be able to:

1. Interpolate a function on a finite-dimensional space.
2. Output solutions using the `DataOut` class.
3. Compute the $L^2$ error of the solution manually.
4. Compute the $L^2$ error using `VectorTools::integrate_difference`.
5. Compare the manual and automated $L^2$ error calculations.
6. Build a graph of the error as a function of the number of degrees of freedom.

## Example Code

Here is an example code snippet that demonstrates these operations:

```cpp
#include <deal.II/base/function_lib.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>
#include <iostream>

using namespace dealii;

int main()
{
  // Create a triangulation and define a finite element space
  Triangulation<2> triangulation;
  FE_Q<2> fe(2);
  DoFHandler<2> dof_handler(triangulation);
  GridGenerator::hyper_cube(triangulation, -2, 2);
  triangulation.refine_global(3);

  dof_handler.distribute_dofs(fe);

  // Interpolate the cosine function on this space
  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> solution_hand_made(dof_handler.n_dofs());
  Functions::CosineFunction<2> cosine;
  VectorTools::interpolate(dof_handler, cosine, solution);

  auto local_support_points = fe.get_unit_support_points();
  MappingQ<2> mapping(fe.degree);
  std::vector<types::global_dof_index> local_to_global(fe.dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    cell->get_dof_indices(local_to_global);
    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
    {
      const auto global_index = local_to_global[i];
      const auto point =
        mapping.transform_unit_to_real_cell(cell, local_support_points[i]);
      solution_hand_made[global_index] = cosine.value(point);
    }
  }

  // First check that we get the same result as deal.II
  auto error = solution;
  error -= solution_hand_made;
  std::cout << "Error of hand made solution: " << error.l2_norm() << std::endl;

  // Now we compute the L2 error
  QGauss<2> quadrature_formula(fe.degree + 1);
  FEValues<2> fe_values(mapping,
                        fe,
                        quadrature_formula,
                        update_values | update_quadrature_points |
                          update_JxW_values);

  double error_L2 = 0;
  std::vector<double> local_values(fe_values.n_quadrature_points);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    double local_cell_error = 0;
    fe_values.get_function_values(solution, local_values);

    for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
    {
      local_cell_error +=
        (local_values[q] - cosine.value(fe_values.quadrature_point(q))) *
        (local_values[q] - cosine.value(fe_values.quadrature_point(q))) *
        fe_values.JxW(q);
    }
    error_L2 += local_cell_error;
  }
  error_L2 = std::sqrt(error_L2);
  std::cout << "Ndofs:  " << dof_handler.n_dofs() << std::endl;
  std::cout << "L2 interpolation error " << error_L2 << std::endl;

  DataOut<2> data_out;
  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;
  data_out.set_flags(flags);
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches(fe.degree);
  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);
}
```

## Exercises

### Exercise 1: Interpolating Functions

1. **Interpolating Different Functions:**
   - Modify the example code to interpolate a sine function instead of a cosine function.
   - Visualize the interpolated solution and compare it with the exact function values.

2. **Higher-Order Functions:**
   - Interpolate a higher-order polynomial function (e.g., a quadratic or cubic function) on the mesh.
   - Visualize the interpolated solution and compare it with the exact function values.

### Exercise 2: Computing Errors Manually

1. **Manual L2 Error Calculation:**
   - Compute the $L^2$ error of the interpolated sine function manually, similar to the example code.
   - Compare the computed error with the exact error.

2. **Comparing Different Functions:**
   - Compute the $L^2$ error for different functions (e.g., exponential, logarithmic) and compare the results.

### Exercise 3: Using VectorTools::integrate_difference

1. **Automated L2 Error Calculation:**
   - Use `VectorTools::integrate_difference` to compute the $L^2$ error for the interpolated sine function.
   - Compare the automated error with the manually computed error.

2. **Error Comparison for Different Functions:**
   - Use `VectorTools::integrate_difference` to compute the $L^2$ error for different functions.
   - Compare the automated errors with the manually computed errors.

### Exercise 4: Visualizing Solutions and Errors

1. **Output and Visualization:**
   - Output the interpolated solutions for different functions to VTU files and visualize them using Paraview.
   - Create visualizations showing the difference between the interpolated and exact solutions.

2. **Error Graph:**
   - Create a graph of the $L^2$ error as a function of the number of degrees of freedom by refining the mesh globally.
   - Analyze the convergence rate of the error with respect to the number of degrees of freedom.
