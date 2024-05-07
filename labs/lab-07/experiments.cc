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
  Triangulation<2> tria;
  GridGenerator::subdivided_hyper_rectangle(tria,
                                            {2, 1},
                                            Point<2>(0, 0),
                                            Point<2>(2, 1));

  (++tria.begin_active())->set_refine_flag();
  tria.execute_coarsening_and_refinement();

  FE_Q<2>       fe(1);
  DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  FunctionParser<2> my_function("2*abs(x-1)", "", "x,y");

  Vector<double> solution(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, my_function, solution);

  MeshWorker::ScratchData<2> scratch_data(fe,
                                          QGauss<2>(2),
                                          update_values | update_gradients |
                                            update_JxW_values,
                                          QGauss<1>(2),
                                          update_values | update_gradients |
                                            update_normal_vectors |
                                            update_JxW_values);
  MeshWorker::CopyData<2>    copy_data;

  auto cell_worker = [](const auto &cell, auto &, auto &) {
    std::cout << "  cell center: " << cell->center() << std::endl;
  };

  auto face_worker = [&](const auto        &cell,
                         const unsigned int face_no,
                         const unsigned int sub_face_no,
                         const auto        &n_cell,
                         const unsigned int n_face_no,
                         const unsigned int n_sub_face_no,
                         auto              &scratch,
                         auto              &copy) {
    auto &fe_v = scratch.reinit(
      cell, face_no, sub_face_no, n_cell, n_face_no, n_sub_face_no);

    std::vector<Tensor<1, 2>> gradients(fe_v.n_quadrature_points);
    fe_v.get_jump_in_function_gradients(solution, gradients);
    double integral = 0;
    for (const auto q : fe_v.quadrature_point_indices())
      {
        integral += gradients[q] * fe_v.normal(q) * fe_v.JxW(q);
      }
    std::cout << "  integral: " << integral << std::endl;
  };

  auto copier = [](const auto &) {};

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
