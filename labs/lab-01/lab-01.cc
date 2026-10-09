#include <deal.II/base/point.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dealii;
namespace fs = std::filesystem;

/** Count the faces on the boundary without managing connectivity arrays. */
unsigned int
count_boundary_faces(const Triangulation<2> &tria)
{
  unsigned int count = 0;
  for (const auto &cell : tria.active_cell_iterators())
    for (unsigned int f = 0; f < cell->n_faces(); ++f)
      if (cell->face(f)->at_boundary())
        ++count;
  return count;
}

/** Show how objects expose vertices and faces. */
void
inspect_cells(const Triangulation<2> &tria)
{
  unsigned int cell_number = 0;
  for (const auto &cell : tria.active_cell_iterators())
    {
      std::cout << "Cell " << cell_number++
                << ", center: " << cell->center() << '\n';
      for (unsigned int v = 0; v < cell->n_vertices(); ++v)
        std::cout << "  vertex " << v << ": " << cell->vertex(v) << '\n';

      for (unsigned int f = 0; f < cell->n_faces(); ++f)
        {
          const auto face = cell->face(f);
          std::cout << "  face " << f
                    << ", boundary: " << face->at_boundary();
          if (face->at_boundary())
            std::cout << ", boundary id: " << face->boundary_id();
          std::cout << '\n';
        }
    }
}

/** Export a geometric mesh (not a finite element solution). */
void
write_mesh(const Triangulation<2> &tria, const fs::path &filename)
{
  fs::create_directories(filename.parent_path());
  std::ofstream output(filename);
  if (!output)
    throw std::runtime_error("Cannot open " + filename.string());
  GridOut grid_out;
  grid_out.write_vtk(tria, output);
}

int
main(int argc, char **argv)
{
  try
    {
      const std::string mode = argc > 1 ? argv[1] : "rectangle";
      if (argc > 2 ||
          (mode != "rectangle" && mode != "shell-flat" &&
           mode != "shell-curved"))
        {
          std::cerr << "Usage: lab-01 [rectangle|shell-flat|shell-curved]\n";
          return 1;
        }

      Triangulation<2> tria;
      if (mode == "rectangle")
        {
          GridGenerator::subdivided_hyper_rectangle(
            tria,
            std::vector<unsigned int>{2, 1},
            Point<2>(0., 0.),
            Point<2>(2., 1.));
          inspect_cells(tria);
        }
      else
        {
          // hyper_shell attaches a SphericalManifold by default.
          GridGenerator::hyper_shell(tria, Point<2>(), 1., 2., 8);
          if (mode == "shell-flat")
            // Retain the coarse mesh but use flat geometry to refine it.
            tria.reset_all_manifolds();
          tria.refine_global(2);
        }

      std::cout << "Active cells: " << tria.n_active_cells()
                << "\nVertices: " << tria.n_vertices()
                << "\nBoundary faces: " << count_boundary_faces(tria)
                << '\n';

      const fs::path output = fs::path("output") / (mode + ".vtk");
      write_mesh(tria, output);
      std::cout << "Wrote " << output << '\n';
    }
  catch (const std::exception &exception)
    {
      std::cerr << "Error: " << exception.what() << '\n';
      return 1;
    }
}
