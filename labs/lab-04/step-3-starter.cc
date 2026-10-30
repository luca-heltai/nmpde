#include <deal.II/base/exceptions.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

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
#include <string>
#include <vector>

using namespace dealii;

// Student starter: the TODOs contain no completed exercise implementation.
// Run from a dedicated directory: mesh.svg and solution.vtu are relative paths.
class Poisson
{
public:
  explicit Poisson(const unsigned int refinements);
  void
  run();

private:
  void
  make_grid();
  double
  integrate_area() const;
  void
  setup_system();
  void
  assemble_system();
  void
  check_assembly() const;
  void
  apply_dirichlet_values();
  void
  solve();
  void
  check_solution() const;
  void
  output_results() const;

  const unsigned int refinements;
  Triangulation<2>   triangulation;
  const FE_Q<2>      fe;
  DoFHandler<2>      dof_handler;
  const MappingQ1<2> mapping;
  const QGauss<2>    quadrature;

  SparsityPattern                           sparsity_pattern;
  SparseMatrix<double>                      system_matrix;
  Vector<double>                            solution;
  Vector<double>                            system_rhs;
  std::map<types::global_dof_index, double> boundary_values;
};

Poisson::Poisson(const unsigned int refinements)
  : refinements(refinements)
  , fe(1)
  , dof_handler(triangulation)
  , quadrature(fe.degree + 1)
{}

void
Poisson::make_grid()
{
  GridGenerator::hyper_cube(triangulation, -1., 1.);
  triangulation.refine_global(refinements);
  dof_handler.distribute_dofs(fe);
  std::cout << "Active cells: " << triangulation.n_active_cells()
            << "\nGlobal DoFs: " << dof_handler.n_dofs() << '\n';

  std::ofstream output("mesh.svg");
  AssertThrow(output, ExcMessage("Cannot open mesh.svg"));
  GridOut().write_svg(triangulation, output);
}

double
Poisson::integrate_area() const
{
  double area = 0.;
  // TODO 1: construct FEValues with mapping, fe, quadrature and the flag
  // needed for JxW. Loop over active cells, reinit on each cell, and sum
  // its quadrature weights in physical coordinates into area.
  return area;
}

void
Poisson::setup_system()
{
  // TODO 2a: construct a DynamicSparsityPattern of size n_dofs(), fill it
  // from the DoFHandler, and copy it to the member sparsity_pattern.
  // Reinitialize system_matrix, solution and system_rhs to the required sizes.
}

void
Poisson::assemble_system()
{
  FEValues<2>        fe_values(mapping,
                        fe,
                        quadrature,
                        update_values | update_gradients | update_JxW_values);
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      (void)cell;
      // TODO 2b: reinit FEValues on this cell and reset both local objects.
      // Use loops over q, i and j to compute the local Poisson matrix and
      // the load vector for f=1. Identify every factor from the weak form.

      // TODO 2c: fill local_dof_indices from this cell; accumulate the
      // local matrix and vector into their global counterparts using +=.
      // SparseMatrix uses add(row, column, value) for accumulation.
    }
}

// These diagnostics are supplied. They do not assemble any entries.
void
Poisson::check_assembly() const
{
  Vector<double> ones(dof_handler.n_dofs());
  ones = 1.;
  Vector<double> image(ones.size());
  system_matrix.vmult(image, ones);

  double symmetry_defect = 0.;
  for (unsigned int row = 0; row < system_matrix.m(); ++row)
    for (auto entry = system_matrix.begin(row); entry != system_matrix.end(row);
         ++entry)
      symmetry_defect =
        std::max(symmetry_defect,
                 std::abs(entry->value() -
                          system_matrix.el(entry->column(), row)));

  double load_sum = 0.;
  for (unsigned int i = 0; i < system_rhs.size(); ++i)
    load_sum += system_rhs[i];

  const double matrix_scale = std::max(1., system_matrix.linfty_norm());
  std::cout << "Before Dirichlet: max |A_ij-A_ji| = " << symmetry_defect
            << "\nBefore Dirichlet: ||A*1||_infty = " << image.linfty_norm()
            << "\nBefore Dirichlet: sum(b) = " << load_sum << '\n';
  AssertThrow(std::abs(load_sum - 4.) <= 1e-12,
              ExcMessage("Load sum must equal the area for f=1 (TODO 2b/2c)."));
  AssertThrow(symmetry_defect <= 1e-12 * matrix_scale,
              ExcMessage("Assembly must preserve symmetry (TODO 2b/2c)."));
  AssertThrow(image.linfty_norm() <= 1e-12 * matrix_scale,
              ExcMessage("Constants must be in the unconstrained kernel."));
  for (unsigned int i = 0; i < system_matrix.m(); ++i)
    AssertThrow(system_matrix.diag_element(i) > 0.,
                ExcMessage("Every assembled diagonal must be positive."));
}

void
Poisson::apply_dirichlet_values()
{
  // TODO 3a: interpolate Functions::ZeroFunction<2>() on boundary id 0
  // into boundary_values. Apply these data to system_matrix, solution and
  // system_rhs with MatrixTools, preserving symmetry by eliminating columns.
}

void
Poisson::solve()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  // TODO 3b: solve the modified system with CG and PreconditionIdentity().
  // Report solver_control.last_step() after the solve.
}

void
Poisson::check_solution() const
{
  Vector<double> residual(solution.size());
  const double   residual_norm =
    system_matrix.residual(residual, solution, system_rhs);
  double boundary_error = 0.;
  for (const auto &entry : boundary_values)
    boundary_error =
      std::max(boundary_error, std::abs(solution[entry.first] - entry.second));

  std::cout << "Boundary DoFs: " << boundary_values.size()
            << "\nResidual ||b-A*U||_2 = " << residual_norm
            << "\nMaximum boundary error = " << boundary_error << '\n';
  AssertThrow(boundary_values.size() == 4u * (1u << refinements),
              ExcMessage("All boundary DoFs must receive Dirichlet data."));
  AssertThrow(residual_norm <= 2e-11 * std::max(1., system_rhs.l2_norm()),
              ExcMessage("The modified linear system is not solved (TODO 3)."));
  AssertThrow(boundary_error <= 1e-12,
              ExcMessage("The prescribed boundary values are not satisfied."));

  // Four unit squares have one free DoF at the origin, with value 3/8.
  if (refinements == 1)
    for (unsigned int i = 0; i < solution.size(); ++i)
      if (boundary_values.find(i) == boundary_values.end())
        {
          std::cout << "Four-cell check: central coefficient = " << solution[i]
                    << " (expected 0.375)\n";
          AssertThrow(std::abs(solution[i] - 0.375) <= 1e-11,
                      ExcMessage(
                        "The four-cell solution must have central value 3/8."));
        }
}

void
Poisson::output_results() const
{
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("solution.vtu");
  AssertThrow(output, ExcMessage("Cannot open solution.vtu"));
  data_out.write_vtu(output);
}

void
Poisson::run()
{
  make_grid();
  const double area = integrate_area();
  std::cout << "Quadrature area: " << area << " (expected 4)\n";
  if (std::abs(area - 4.) > 1e-12)
    {
      std::cout << "[TODO 1] Complete the area integral before proceeding.\n";
      return;
    }

  setup_system();
  if (system_matrix.m() != dof_handler.n_dofs() ||
      solution.size() != dof_handler.n_dofs() ||
      system_rhs.size() != dof_handler.n_dofs())
    {
      std::cout
        << "[TODO 2a] Allocate the matrix and vectors before assembly.\n";
      return;
    }

  assemble_system();
  check_assembly();
  apply_dirichlet_values();
  if (boundary_values.empty())
    {
      std::cout << "[TODO 3a] Impose Dirichlet data before calling CG.\n";
      return;
    }

  solve();
  check_solution();
  output_results();
  std::cout << "Checks passed. Wrote solution.vtu.\n";
}

int
main(int argc, char *argv[])
{
  try
    {
      AssertThrow(argc <= 2, ExcMessage("Usage: lab-04 [refinements: 1..5]"));
      unsigned int refinements = 3;
      if (argc == 2)
        {
          const std::string argument(argv[1]);
          AssertThrow(
            argument.size() == 1 && argument[0] >= '1' && argument[0] <= '5',
            ExcMessage("Refinements must be an integer from 1 to 5."));
          refinements = static_cast<unsigned int>(argument[0] - '0');
        }
      std::cout << std::setprecision(12);
      Poisson(refinements).run();
    }
  catch (const std::exception &exception)
    {
      std::cerr << exception.what() << '\n';
      return 1;
    }
  return 0;
}
