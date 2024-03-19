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

/**
 * Aim for this lab:
 *
 * 1. Create a triangulation
 * 2. Create a DoFHandler
 * 3. Create a finite element
 * 4. Distribute the degrees of freedom
 * 5. Interpolate a Function on the finite dimensional space
 * 6. Output the solution (using DataOut)
 * 7. Compute the error of the solution manually
 * 8. Output the error
 * 9. Compute the error of the solution using VectorTools::integrate_difference
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
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global();

  // See step-2/step-3
  dof_handler.distribute_dofs(fe);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "Total number of cells: " << triangulation.n_cells()
            << std::endl;

  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  std::cout << "Number of dofs per cell: " << fe.dofs_per_cell << std::endl;

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

  DataOut<2> data_out;

  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;
  data_out.set_flags(flags);

  data_out.attach_dof_handler(dof_handler);

  std::vector<Vector<double>> basis_functions(
    10, Vector<double>(dof_handler.n_dofs()));
  for (unsigned int i = 0; i < 10; ++i)
    {
      Vector<double> &basis_function = basis_functions[i];
      basis_function[i]              = 1;
      data_out.add_data_vector(basis_function,
                               "basis_function_" + std::to_string(i));
    }
  data_out.build_patches(fe.degree);
  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);
}