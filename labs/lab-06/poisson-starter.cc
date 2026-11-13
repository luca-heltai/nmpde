#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

#include "manufactured_functions.h"

using namespace dealii;

// The assembly and linear solver are supplied. Complete the three TODOs.
void
run(const Function<2> &exact,
    const unsigned int refinement,
    std::ostream      &errors)
{
  Triangulation<2> mesh;
  GridGenerator::hyper_cube(mesh, -1., 1.);
  mesh.refine_global(refinement);
  FE_Q<2>       fe(1);
  DoFHandler<2> dofs(mesh);
  dofs.distribute_dofs(fe);
  DynamicSparsityPattern dsp(dofs.n_dofs());
  DoFTools::make_sparsity_pattern(dofs, dsp);
  SparsityPattern sparsity;
  sparsity.copy_from(dsp);
  SparseMatrix<double> matrix(sparsity);
  Vector<double>       rhs(dofs.n_dofs()), solution(dofs.n_dofs());

  QGauss<2>          quadrature(fe.degree + 3);
  FEValues<2>        values(fe,
                     quadrature,
                     update_values | update_gradients |
                       update_quadrature_points | update_JxW_values);
  FullMatrix<double> cell_matrix(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
  Vector<double>     cell_rhs(fe.n_dofs_per_cell());
  std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
  for (const auto &cell : dofs.active_cell_iterators())
    {
      values.reinit(cell);
      cell_matrix = 0.;
      cell_rhs    = 0.;
      for (const auto q : values.quadrature_point_indices())
        {
          // TODO 1: replace f=0 with minus the Laplacian of exact at x_q.
          // Function::laplacian is virtual; the chosen function implements it.
          const double f = 0.;
          for (const auto i : values.dof_indices())
            {
              cell_rhs(i) += f * values.shape_value(i, q) * values.JxW(q);
              for (const auto j : values.dof_indices())
                cell_matrix(i, j) += values.shape_grad(i, q) *
                                     values.shape_grad(j, q) * values.JxW(q);
            }
        }
      cell->get_dof_indices(indices);
      for (const auto i : values.dof_indices())
        {
          rhs[indices[i]] += cell_rhs(i);
          for (const auto j : values.dof_indices())
            matrix.add(indices[i], indices[j], cell_matrix(i, j));
        }
    }

  std::map<types::global_dof_index, double> boundary;
  // TODO 2: replace ZeroFunction with the SAME exact function used above.
  VectorTools::interpolate_boundary_values(dofs,
                                           0,
                                           Functions::ZeroFunction<2>(),
                                           boundary);
  // Eliminating columns also updates the RHS and preserves symmetry for CG.
  MatrixTools::apply_boundary_values(boundary, matrix, solution, rhs, true);
  SolverControl            control(4000, 1e-12);
  SolverCG<Vector<double>> solver(control);
  solver.solve(matrix, solution, rhs, PreconditionIdentity());
  Vector<double> residual(solution.size());
  const double   norm           = matrix.residual(residual, solution, rhs);
  double         boundary_error = 0.;
  std::map<types::global_dof_index, double> expected_boundary;
  VectorTools::interpolate_boundary_values(dofs, 0, exact, expected_boundary);
  for (const auto &entry : expected_boundary)
    boundary_error =
      std::max(boundary_error, std::abs(solution[entry.first] - entry.second));
  std::cout << "refinement=" << refinement << " cells=" << mesh.n_active_cells()
            << " dofs=" << dofs.n_dofs() << " residual=" << norm
            << " boundary error=" << boundary_error << '\n';
  AssertThrow(norm < 1e-10 * std::max(1., rhs.l2_norm()),
              ExcMessage("Large residual"));

  DataOut<2> output;
  output.attach_dof_handler(dofs);
  output.add_data_vector(solution, "solution");
  Vector<double> exact_nodes(solution.size());
  VectorTools::interpolate(dofs, exact, exact_nodes);
  output.add_data_vector(exact_nodes, "exact_at_nodes");
  output.build_patches();
  std::ofstream vtu("solution-" + std::to_string(refinement) + ".vtu");
  AssertThrow(vtu, ExcMessage("Cannot open VTU file"));
  output.write_vtu(vtu);

  // TODO 3: compute L2_norm and H1_norm errors with integrate_difference
  // and compute_global_error, using QGauss<2>(fe.degree+3).
  // Append one CSV row: degree,h,l2,h1. Here h=2./(1u<<refinement).
  // Repeat with one more quadrature point in each direction to check accuracy.
  (void)errors;
}

int
main()
{
  try
    {
      std::cout << std::setprecision(12);
      std::ofstream errors("errors.csv");
      AssertThrow(errors, ExcMessage("Cannot open errors.csv"));
      errors << std::setprecision(16) << "degree,h,l2,h1\n";
      Lab06::Affine exact; // Exercise 2: Functions::ExpFunction<2>; exercise 3:
                           // Lab06::TwoPeaks.
      // First run refinement 2 alone. Then use refinements 2,...,6.
      for (unsigned int refinement = 2; refinement <= 2; ++refinement)
        run(exact, refinement, errors);
    }
  catch (const std::exception &error)
    {
      std::cerr << error.what() << '\n';
      return 1;
    }
}
