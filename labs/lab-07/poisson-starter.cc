#include <deal.II/base/function_parser.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace dealii;

class Poisson
{
public:
  explicit Poisson(const std::string &parameter_file);
  void
  run();

private:
  void
  setup_system();
  void
  assemble_system();
  void
  solve();
  void
  refine_grid();

  unsigned int           initial_refinement = 2;
  unsigned int           n_cycles           = 4;
  FunctionParser<2>      exact_solution;
  FunctionParser<2>      forcing;
  ParsedConvergenceTable error_table;

  Triangulation<2>     triangulation;
  const FE_Q<2>        fe{1};
  DoFHandler<2>        dof_handler{triangulation};
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double>       solution;
  Vector<double>       system_rhs;
};

Poisson::Poisson(const std::string &parameter_file)
{
  ParameterHandler prm;
  std::string      exact_expression   = "x + 2*y + sin(pi*x)*sin(pi*y)";
  std::string      forcing_expression = "2*pi^2*sin(pi*x)*sin(pi*y)";

  prm.enter_subsection("Poisson");
  prm.add_parameter("Initial refinement", initial_refinement);
  prm.add_parameter("Number of cycles", n_cycles, "", Patterns::Integer(1));
  prm.add_parameter("Exact solution", exact_expression);
  prm.add_parameter("Right hand side", forcing_expression);
  prm.leave_subsection();
  prm.enter_subsection("Convergence table");
  error_table.add_parameters(prm);
  prm.leave_subsection();
  prm.parse_input(parameter_file);

  const std::map<std::string, double> constants{{"pi", numbers::PI}};
  exact_solution.initialize("x,y", exact_expression, constants);
  forcing.initialize("x,y", forcing_expression, constants);
}

void
Poisson::setup_system()
{
  dof_handler.distribute_dofs(fe);
  const auto n_dofs = dof_handler.n_dofs();

  DynamicSparsityPattern dynamic_pattern(n_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_pattern);
  system_matrix.clear();
  sparsity_pattern.copy_from(dynamic_pattern);
  system_matrix.reinit(sparsity_pattern);
  solution.reinit(n_dofs);
  system_rhs.reinit(n_dofs);
}

void
Poisson::assemble_system()
{
  const QGauss<2>                      quadrature(fe.degree + 3);
  FEValues<2>                          values(fe,
                     quadrature,
                     update_values | update_gradients |
                       update_quadrature_points | update_JxW_values);
  const unsigned int                   n_local = fe.n_dofs_per_cell();
  FullMatrix<double>                   cell_matrix(n_local, n_local);
  Vector<double>                       cell_rhs(n_local);
  std::vector<types::global_dof_index> indices(n_local);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      values.reinit(cell);
      cell_matrix = 0.;
      cell_rhs    = 0.;
      for (const auto q : values.quadrature_point_indices())
        {
          const double f = forcing.value(values.quadrature_point(q));
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
          system_rhs(indices[i]) += cell_rhs(i);
          for (const auto j : values.dof_indices())
            system_matrix.add(indices[i], indices[j], cell_matrix(i, j));
        }
    }

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           exact_solution,
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
}

void
Poisson::solve()
{
  SolverControl            control(4000, 1e-12);
  SolverCG<Vector<double>> solver(control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
}

void
Poisson::refine_grid()
{
  triangulation.refine_global(1);
}

void
Poisson::run()
{
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(initial_refinement);

  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      if (cycle > 0)
        refine_grid();
      setup_system();
      assemble_system();
      solve();
      error_table.error_from_exact(dof_handler, solution, exact_solution);
    }

  error_table.output_table(std::cout);
}

int
main(int argc, char **argv)
{
  if (argc != 2)
    {
      std::cerr << "Usage: lab-07 path/to/poisson.prm\n";
      return 1;
    }

  try
    {
      Poisson problem(argv[1]);
      problem.run();
    }
  catch (const std::exception &error)
    {
      std::cerr << error.what() << '\n';
      return 1;
    }
}
