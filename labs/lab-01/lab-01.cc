#include <deal.II/base/point.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <iostream>

using namespace dealii;

int
main()
{
  // Create the center of a circle
  Point<2>         center;
  double           inner_radius = 1.0;
  double           outer_radius = 2.0;
  Triangulation<2> triangulation;
  FE_Q<2>          fe(2);
  DoFHandler<2>    dof_handler(triangulation);

  GridGenerator::hyper_shell(triangulation, center, inner_radius, outer_radius);

  // triangulation.refine_global();

  // See step-2/step-3
  dof_handler.distribute_dofs(fe);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "Total number of cells: " << triangulation.n_cells()
            << std::endl;

  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
  GridOut grid_out;

  std::string   filename = "grid-1.svg";
  std::ofstream out(filename);
  grid_out.write_svg(triangulation, out);
}