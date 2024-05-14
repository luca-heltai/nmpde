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
  // Define the center of the circle
  Point<2> center;

  // Define inner and outer radii of the hyper shell
  double inner_radius = 1.0;
  double outer_radius = 2.0;

  // Create a 2D triangulation object
  Triangulation<2> triangulation;

  // Define a finite element space with polynomial degree 2
  FE_Q<2> fe(2);

  // Set up the DoF handler with the triangulation
  DoFHandler<2> dof_handler(triangulation);

  // Generate a hyper shell mesh (annulus) with specified radii
  GridGenerator::hyper_shell(triangulation, center, inner_radius, outer_radius);

  // Refine the mesh globally (optional, uncomment this if needed)
  // triangulation.refine_global();

  // Distribute degrees of freedom among the cells (see step-2/step-3 tutorials
  // for more details)
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

  // Define the output file name
  std::string filename = "grid-1.svg";

  // Create an output file stream and write the grid to the file in SVG format
  std::ofstream out(filename);
  grid_out.write_svg(triangulation, out);
}
