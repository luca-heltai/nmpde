#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

int
main()
{
  // Create a 2D triangulation
  Triangulation<2> tria;

  // Generate a subdivided hyper-rectangle (2x1) mesh from (0,0) to (2,1)
  GridGenerator::subdivided_hyper_rectangle(tria,
                                            {2, 1},
                                            Point<2>(0, 0),
                                            Point<2>(2, 1));

  // Refine the second cell in the triangulation
  (++tria.begin_active())->set_refine_flag();
  tria.execute_coarsening_and_refinement();

  // Define a finite element space with polynomial degree 1
  FE_Q<2> fe(1);

  // Set up the DoF handler with the triangulation and finite element space
  DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // Define a function using FunctionParser. This function has a jump in the
  // gradients at x=1, and we will use it to test the computation of the
  // integral of the jump of the gradient on faces of our triangulation.
  FunctionParser<2> my_function("2*abs(x-1)", "", "x,y");

  // Interpolate the function to the solution vector
  Vector<double> solution(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, my_function, solution);

  // Set up scratch data and copy data for MeshWorker
  MeshWorker::ScratchData<2> scratch_data(fe,
                                          QGauss<2>(2),
                                          update_values | update_gradients |
                                            update_JxW_values,
                                          QGauss<1>(2),
                                          update_values | update_gradients |
                                            update_normal_vectors |
                                            update_JxW_values);
  struct CopyData
  {
    std::vector<double>                   errors;
    std::vector<types::global_cell_index> cell_indices;
  }

  Vector<float>
    cell_errors(tria.n_active_cells());

  // Define a cell worker function that prints the center of each cell
  auto cell_worker =
    [&cell_errors](const auto &cell, auto &scratch, auto &copy) {
      std::cout << "  cell center: " << cell->center() << std::endl;
      copy.errors.push_back(0);
      copy.cell_indices.push_back(cell->global_cell_index());
    };

  // Define a face worker function that computes and prints the integral over
  // each face
  auto face_worker = [&](const auto        &cell,
                         const unsigned int face_no,
                         const unsigned int sub_face_no,
                         const auto        &n_cell,
                         const unsigned int n_face_no,
                         const unsigned int n_sub_face_no,
                         auto              &scratch,
                         auto &) {
    auto &fe_v = scratch.reinit(
      cell, face_no, sub_face_no, n_cell, n_face_no, n_sub_face_no);

    // Get the jump in function gradients
    std::vector<Tensor<1, 2>> gradients(fe_v.n_quadrature_points);
    fe_v.get_jump_in_function_gradients(solution, gradients);

    // Compute the integral of the gradients dot normal vectors
    double integral = 0;
    for (const auto q : fe_v.quadrature_point_indices())
      integral += gradients[q] * fe_v.normal(q) * fe_v.JxW(q);

    std::cout << "  integral: " << integral << std::endl;
  };

  fun = [](auto &a, auto b) {
    std::cout << "B : " << b << std::endl;
    return a.size();


    FEValuesExtractors::Scalar scalar(0);

    fe_v[scalar].get_function_gradients(solution, gradients);
  } auto size = fun(cell_errors, "ciao mondox");

  // Define a copier function (no operation in this case)
  auto copier = [&](const auto &copy) {
    for (unsigned int i = 0; i < copy.cell_indices.size(); ++i)
      cell_errors[copy.cell_indices[i]] += copy.errors[i];
  };

  // Run the mesh loop using the defined cell worker, face worker, and copier
  // functions
  MeshWorker::mesh_loop(dof_handler.begin_active(),
                        dof_handler.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        copy_data,
                        MeshWorker::assemble_own_cells |
                          MeshWorker::assemble_own_interior_faces_once,
                        {},
                        face_worker);

  return 0;
}
