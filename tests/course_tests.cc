#include <gtest/gtest.h>

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
  write_mesh_svg(const fs::path &filename, const unsigned int refinement)
  {
    fs::create_directories(filename.parent_path());

    dealii::Triangulation<2> triangulation;
    dealii::GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
    triangulation.refine_global(refinement);

    std::ofstream output(filename);
    if (!output)
      return false;

    dealii::GridOut grid_out;
    grid_out.write_svg(triangulation, output);
    output.close();

    return fs::is_regular_file(filename) && fs::file_size(filename) > 0;
  }
} // namespace

TEST(CourseFiles, LaboratorySourcesArePresent)
{
  const std::vector<std::string> labs = {
    "lab-01", "lab-02", "lab-03", "lab-04", "lab-05", "lab-05b",
    "lab-05c", "lab-05d", "lab-05e", "lab-06", "lab-07", "lab-08",
    "lab-09"};

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

TEST(CourseAssets, GenerateLectureMeshFigure)
{
  EXPECT_TRUE(write_mesh_svg(asset_root / "lecture-01-poisson-mesh.svg", 2));
}

TEST(CourseAssets, GenerateLaboratoryMeshFigure)
{
  EXPECT_TRUE(write_mesh_svg(asset_root / "lab-01-refined-mesh.svg", 3));
}
