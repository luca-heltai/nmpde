#include <gtest/gtest.h>

#include <deal.II/base/point.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
  const fs::path source_root = NMPDE_SOURCE_DIR;
  const fs::path asset_root  = NMPDE_ASSET_DIR;

  bool
  write_mesh_svg(const dealii::Triangulation<2> &tria,
                 const fs::path                 &filename)
  {
    fs::create_directories(filename.parent_path());
    std::ofstream output(filename);
    if (!output)
      return false;

    dealii::GridOut grid_out;
    grid_out.write_svg(tria, output);
    output.close();
    return fs::is_regular_file(filename) && fs::file_size(filename) > 0;
  }

  unsigned int
  count_boundary_faces(const dealii::Triangulation<2> &tria)
  {
    unsigned int count = 0;
    for (const auto &cell : tria.active_cell_iterators())
      for (unsigned int f = 0; f < cell->n_faces(); ++f)
        if (cell->face(f)->at_boundary())
          ++count;
    return count;
  }
} // namespace

TEST(CourseFiles, LaboratorySourcesArePresent)
{
  const std::vector<std::string> labs = {
    "lab-01", "lab-02", "lab-03", "lab-04", "lab-05", "lab-06",
    "lab-06b", "lab-06c", "lab-06d", "lab-06e", "lab-07", "lab-08",
    "lab-09", "lab-10"};

  for (const auto &lab : labs)
    {
      EXPECT_TRUE(fs::is_regular_file(source_root / "labs" / lab /
                                      (lab + ".cc")))
        << "Missing C++ source for " << lab;
      EXPECT_TRUE(fs::is_regular_file(source_root / "labs" / lab /
                                      "README.md"))
        << "Missing README for " << lab;
    }
}

TEST(CourseFiles, LegacyVtkExampleIsPresent)
{
  EXPECT_TRUE(fs::is_regular_file(
    source_root / "labs/lab-02/data/two-quads.vtk"));
}

TEST(CourseGeometry, TwoQuadrilateralsShareOneFace)
{
  dealii::Triangulation<2> tria;
  dealii::GridGenerator::subdivided_hyper_rectangle(
    tria,
    std::vector<unsigned int>{2, 1},
    dealii::Point<2>(0., 0.),
    dealii::Point<2>(2., 1.));

  EXPECT_EQ(tria.n_active_cells(), 2);
  EXPECT_EQ(tria.n_vertices(), 6);
  EXPECT_EQ(count_boundary_faces(tria), 6);

  unsigned int incidences = 0;
  for (const auto &cell : tria.active_cell_iterators())
    incidences += cell->n_faces();
  EXPECT_EQ(incidences, 8);
}

TEST(CourseAssets, GenerateLectureMeshFigure)
{
  dealii::Triangulation<2> tria;
  dealii::GridGenerator::hyper_shell(
    tria, dealii::Point<2>(), 1., 2.);
  tria.refine_global(2);
  EXPECT_TRUE(write_mesh_svg(tria,
                             asset_root / "lecture-01-poisson-mesh.svg"));
}

TEST(CourseAssets, GenerateLaboratoryMeshFigures)
{
  dealii::Triangulation<2> rectangle;
  dealii::GridGenerator::subdivided_hyper_rectangle(
    rectangle,
    std::vector<unsigned int>{2, 1},
    dealii::Point<2>(0., 0.),
    dealii::Point<2>(2., 1.));
  EXPECT_TRUE(write_mesh_svg(rectangle,
                             asset_root / "lab-02-two-quads.svg"));

  dealii::Triangulation<2> flat;
  dealii::GridGenerator::hyper_shell(flat, dealii::Point<2>(), 1., 2., 8);
  flat.reset_all_manifolds();
  flat.refine_global(2);
  EXPECT_TRUE(write_mesh_svg(flat,
                             asset_root / "lab-02-flat-shell.svg"));

  dealii::Triangulation<2> curved;
  dealii::GridGenerator::hyper_shell(curved, dealii::Point<2>(), 1., 2., 8);
  curved.refine_global(2);
  EXPECT_TRUE(write_mesh_svg(curved,
                             asset_root / "lab-02-curved-shell.svg"));
}
