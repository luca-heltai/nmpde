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

/**
 * Aim for this lab:
 *
 * 1. Create a triangulation
 * 2. Create a DoFHandler
 * 3. Create a finite element
 * 4. Distribute the degrees of freedom
 * 5. Interpolate a Function on the finite dimensional space
 * 6. Output the solution (using DataOut)
 * 7. Compute the $L^2$ error of the solution manually (looping over cells)
 * 8. Output the error
 * 9. Compute the $L^2$ error of the solution using
 * VectorTools::integrate_difference
 * 10. Check that the two coincide!
 * 11. Plug everything inside a function, and build a graph of the error as a
 * function of the number of degrees of freedom, refining the triangulation a
 * few times globally
 */

int
main()
{
  Triangulation<2> triangulation;
  FE_Q<2>          fe(2);
  DoFHandler<2>    dof_handler(triangulation);
  GridGenerator::hyper_cube(triangulation, -2, 2);
  triangulation.refine_global(3);

  dof_handler.distribute_dofs(fe);

  // Interpolate the cosine function on this space.
  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> solution_hand_made(dof_handler.n_dofs());
  // Use interpolate to fill the solution vector with the values of the
  // function at the degrees of freedom.
  Functions::CosineFunction<2> cosine;

  VectorTools::interpolate(dof_handler, cosine, solution);

  // We fill solution_hand_made with the correct values of the function.
  auto local_support_points = fe.get_unit_support_points();

  // Mapping from reference to real cell
  MappingQ<2> mapping(fe.degree);

  // Global numbering of degrees of freedom.
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

  // First check that we get the same result of deal.II
  auto error = solution;
  error -= solution_hand_made;
  std::cout << "Error of hand made solution: " << error.l2_norm() << std::endl;

  // Now we compute the L2 error
  QGauss<2> quadrature_formula(fe.degree + 1);

  // Construct a FEValues object to evaluate the shape functions and the
  // Jacobian of the transformation from the reference cell to the real cell.
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
      // Computes v[q] = sum_i u_[local_to_global[i]] phi_i(q)
      fe_values.get_function_values(solution, local_values);

      for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
        {
          // compute difference of the two solutions at the quadrature point
          // int_{T} (u_h - u)^2 dx
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