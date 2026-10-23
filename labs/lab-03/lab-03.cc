#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

using namespace dealii;

// This file is the student starter. TODOs intentionally contain no solution.
// Use a separate run directory; output filenames are relative to that
// directory.

void
make_grid(Triangulation<2> &triangulation)
{
  GridGenerator::subdivided_hyper_rectangle(triangulation,
                                            {2, 1},
                                            Point<2>(0., 0.),
                                            Point<2>(2., 1.));
  std::ofstream mesh_file("mesh.svg");
  AssertThrow(mesh_file, ExcMessage("Cannot open mesh.svg"));
  GridOut().write_svg(triangulation, mesh_file);
}

void
inspect_topology(const Triangulation<2> &triangulation)
{
  // TODO 1a (Exercise 1): traverse active cells. For every cell print its
  // level/index, local vertex -> mesh vertex index, and local face -> mesh
  // face index. Also print vertex coordinates and face endpoint vertex IDs.
  // Count UNIQUE mesh vertices and faces using sets, not cell incidences.
  (void)triangulation;
  std::cout << "[TODO 1a] Mesh-object inspection is not implemented.\n";
}

void
inspect_element(const FiniteElement<2> &fe)
{
  // TODO 1b (Exercise 1): print the three EXCLUSIVE dpo entries and the
  // INCLUSIVE number of DoFs per face and per cell. Compare with your
  // prediction sheet before running the program.
  (void)fe;
  std::cout << "[TODO 1b] Finite-element inspection is not implemented.\n";
}

void
inspect_dofs(const DoFHandler<2> &dof_handler)
{
  // TODO 2 (Exercise 2): print (cell, local basis index, global DoF index).
  // Size the index vector BEFORE calling get_dof_indices().
  // Store each cell's global DoF IDs in a set and find their intersection.
  // Compare the same mesh vertex's vertex_index() and vertex_dof_index()
  // for FE_Q only. FE_DGQ has no DoFs attached to vertices.
  (void)dof_handler;
  std::cout << "[TODO 2] Local-to-global inspection is not implemented.\n";
}

// Output plumbing is supplied. This function neither selects coefficients
// nor chooses a basis function: these are student tasks in Exercise 3.
void
write_field(const DoFHandler<2>  &dof_handler,
            const Vector<double> &coefficients,
            const std::string    &filename)
{
  AssertThrow(coefficients.size() == dof_handler.n_dofs(),
              ExcMessage("Coefficient vector has the wrong size"));
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(coefficients, "field");
  data_out.build_patches(dof_handler.get_fe().degree);
  std::ofstream output(filename);
  AssertThrow(output, ExcMessage("Cannot open " + filename));
  data_out.write_vtu(output);
}

void
renumber_and_visualize(DoFHandler<2> &dof_handler, const std::string &tag)
{
  // TODO 3a (Exercise 3): select one global basis function; construct its
  // coefficient vector and call write_field() with tag + "-before.vtu".
  // Save topology and local-to-global tables BEFORE changing the numbering.
  //
  // TODO 3b: construct a reverse permutation in a vector of global_dof_index.
  // Check it is a bijection, then call renumber_dofs(). Read its contract:
  // each entry stores the NEW number of the DoF with the given OLD number.
  // Reorder the coefficients consistently and write tag + "-after.vtu".
  // Verify the local coefficient lists agree before/after, and reprint
  // topology and DoF tables. Do not assume renumber_dofs() changes vectors.
  (void)dof_handler;
  (void)tag;
  std::cout << "[TODO 3] Renumbering and field output are not implemented.\n";
}

void
run_case(Triangulation<2>       &triangulation,
         const FiniteElement<2> &fe,
         const std::string      &tag)
{
  std::cout << "\n=== " << tag << " : " << fe.get_name() << " ===\n";
  inspect_element(fe);
  DoFHandler<2> dof_handler(triangulation);

  // TODO 1c (Exercise 1): distribute the DoFs using the supplied element fe.

  if (dof_handler.n_dofs() == 0)
    {
      std::cout << "[TODO 1c] No DoFs distributed; later tasks skipped.\n";
      return;
    }

  std::cout << "Global DoFs: " << dof_handler.n_dofs() << '\n';
  inspect_dofs(dof_handler);
  renumber_and_visualize(dof_handler, tag);
}

int
main()
{
  try
    {
      Triangulation<2> triangulation;
      make_grid(triangulation);
      inspect_topology(triangulation);

      const FE_Q<2>   q1(1);
      const FE_Q<2>   q2(2);
      const FE_DGQ<2> dgq2(2);
      run_case(triangulation, q1, "q1");
      run_case(triangulation, q2, "q2");
      run_case(triangulation, dgq2, "dgq2");
    }
  catch (const std::exception &exception)
    {
      std::cerr << exception.what() << '\n';
      return 1;
    }
  return 0;
}
