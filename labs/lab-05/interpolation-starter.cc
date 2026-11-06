#include <deal.II/base/exceptions.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dealii;

// Complete one case first, then enable the convergence loops in main().
void run_case(const unsigned int degree,
              const unsigned int refinement,
              std::ostream &csv)
{
  Triangulation<2> triangulation;
  GridGenerator::hyper_cube(triangulation, -1., 1.);
  triangulation.refine_global(refinement);
  FE_Q<2> fe(degree);
  DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  MappingQ1<2> mapping;
  Functions::CosineFunction<2> exact;
  Vector<double> interpolated(dof_handler.n_dofs());

  // TODO 1: fill interpolated with VectorTools::interpolate.
  // Remove the following exception after completing this step.
  throw std::runtime_error("TODO 1: interpolate the exact function");

  // The cosine is not polynomial: check the result also with degree + 5.
  QGauss<2> quadrature(degree + 3);
  FEValues<2> fe_values(mapping, fe, quadrature,
                        update_values | update_quadrature_points |
                          update_JxW_values);
  std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
  std::vector<double> manual_values(quadrature.size());
  std::vector<double> library_values(quadrature.size());
  double squared_error = 0.;
  double max_value_difference = 0.;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell->get_dof_indices(indices);
      std::fill(manual_values.begin(), manual_values.end(), 0.);

      // TODO 2: for every q, sum interpolated[indices[i]] * shape_value(i,q).
      // This evaluates the local restriction of the global interpolant.
      throw std::runtime_error("TODO 2: evaluate the interpolant manually");

      // TODO 3: fill library_values with get_function_values and compare
      // all quadrature values, updating max_value_difference.
      throw std::runtime_error("TODO 3: compare with get_function_values");

      // TODO 4: accumulate the squared L2 error using manual_values,
      // exact.value(quadrature_point(q)), and JxW(q).
      throw std::runtime_error("TODO 4: integrate the squared difference");
    }

  AssertThrow(max_value_difference < 1.e-12,
              ExcMessage("Manual and library evaluations disagree"));
  const double error_l2 = std::sqrt(squared_error);
  // h is the side length of a cell; its diameter is sqrt(2)*h.
  const double h = 2. / std::pow(2., refinement);
  csv << degree << ',' << refinement << ',' << h << ','
      << dof_handler.n_dofs() << ',' << error_l2 << '\n';
  std::cout << "p=" << degree << " r=" << refinement
            << " L2=" << error_l2
            << " max evaluation difference=" << max_value_difference << '\n';

  // TODO 6 (after the convergence experiment): compare error_l2 with
  // integrate_difference + compute_global_error using the SAME quadrature.
  // Then compute H1_norm and append an h1 column to the CSV header and rows.
}

int main()
{
  try
    {
      std::ofstream csv("errors.csv");
      AssertThrow(csv.is_open(), ExcMessage("Cannot open errors.csv"));
      csv << std::setprecision(17);
      csv << "degree,refinement,h,dofs,l2\n";
      // TODO 5: replace this call with loops over p=1,2,3 and r=1,...,6.
      run_case(1, 2, csv);
    }
  catch (const std::exception &error)
    {
      std::cerr << error.what() << '\n';
      return 1;
    }
}
