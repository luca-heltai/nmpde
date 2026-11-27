#include <deal.II/base/function_parser.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace dealii;

constexpr types::boundary_id dirichlet_id = 0;
constexpr types::boundary_id neumann_id   = 1;
constexpr unsigned int       whole_face   = numbers::invalid_unsigned_int;

double
squared_normal_jump(const FEInterfaceValues<2> &interface,
                    const Vector<double>       &solution)
{
  // TODO 2: Read both gradient traces through FEInterfaceValues and
  // integrate the square of their difference dotted with interface.normal(q).
  (void)interface;
  (void)solution;
  return 0.;
}

class Poisson
{
public:
  explicit Poisson(const std::string &parameter_file);
  void
  run();


private:
  void
  setup_system();
  void
  assemble_system();
  void
  solve();
  void
  estimate();
  void
  add_face_residuals(Vector<double> &squared_indicators) const;
  bool
  mark();
  void
  output_solution(unsigned int cycle) const;

  unsigned int           initial_refinement = 2;
  unsigned int           n_cycles           = 4;
  std::string            refinement         = "mean-square";
  bool                   mixed_boundary     = false;
  bool                   use_face_residuals = false;
  FunctionParser<2>      exact_solution;
  FunctionParser<2>      forcing;
  FunctionParser<2>      neumann_data;
  ParsedConvergenceTable error_table;

  Triangulation<2>          triangulation;
  const FE_Q<2>             fe{1};
  DoFHandler<2>             dof_handler{triangulation};
  AffineConstraints<double> constraints;
  SparsityPattern           sparsity_pattern;
  SparseMatrix<double>      system_matrix;
  Vector<double>            solution;
  Vector<double>            system_rhs;
  Vector<double>            estimator;
  Vector<double>            squared_cell_norms;
};

Poisson::Poisson(const std::string &parameter_file)
{
  ParameterHandler prm;
  std::string      exact_expression   = "x + 2*y + sin(pi*x)*sin(pi*y)";
  std::string      forcing_expression = "2*pi^2*sin(pi*x)*sin(pi*y)";

  std::string neumann_expression = "0";

  prm.enter_subsection("Poisson");
  prm.add_parameter("Initial refinement", initial_refinement);
  prm.add_parameter("Number of cycles", n_cycles, "", Patterns::Integer(1));
  prm.add_parameter("Exact solution", exact_expression);
  prm.add_parameter("Right hand side", forcing_expression);
  prm.add_parameter("Refinement",
                    refinement,
                    "",
                    Patterns::Selection("uniform|norm-over-cells|mean-square"));
  prm.add_parameter("Mixed boundary", mixed_boundary);
  prm.add_parameter("Neumann data", neumann_expression);
  prm.add_parameter("Use face residuals", use_face_residuals);
  prm.leave_subsection();
  prm.enter_subsection("Convergence table");
  error_table.add_parameters(prm);
  prm.leave_subsection();
  prm.parse_input(parameter_file);

  const std::map<std::string, double> constants{{"pi", numbers::PI}};
  exact_solution.initialize("x,y", exact_expression, constants);
  forcing.initialize("x,y", forcing_expression, constants);
  neumann_data.initialize("x,y", neumann_expression, constants);
}

void
Poisson::setup_system()
{
  dof_handler.distribute_dofs(fe);
  const auto n_dofs = dof_handler.n_dofs();

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           dirichlet_id,
                                           exact_solution,
                                           constraints);
  constraints.close();

  DynamicSparsityPattern dynamic_pattern(n_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_pattern, constraints);
  system_matrix.clear();
  sparsity_pattern.copy_from(dynamic_pattern);
  system_matrix.reinit(sparsity_pattern);
  solution.reinit(n_dofs);
  system_rhs.reinit(n_dofs);
  estimator.reinit(triangulation.n_active_cells());
  squared_cell_norms.reinit(triangulation.n_active_cells());
}

void
Poisson::assemble_system()
{
  const QGauss<2>                      quadrature(fe.degree + 3);
  FEValues<2>                          values(fe,
                     quadrature,
                     update_values | update_gradients |
                       update_quadrature_points | update_JxW_values);
  const unsigned int                   n_local = fe.n_dofs_per_cell();
  FullMatrix<double>                   cell_matrix(n_local, n_local);
  Vector<double>                       cell_rhs(n_local);
  std::vector<types::global_dof_index> indices(n_local);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      values.reinit(cell);
      cell_matrix = 0.;
      cell_rhs    = 0.;
      for (const auto q : values.quadrature_point_indices())
        {
          const double f = forcing.value(values.quadrature_point(q));
          for (const auto i : values.dof_indices())
            {
              cell_rhs(i) += f * values.shape_value(i, q) * values.JxW(q);
              for (const auto j : values.dof_indices())
                cell_matrix(i, j) += values.shape_grad(i, q) *
                                     values.shape_grad(j, q) * values.JxW(q);
            }
        }

      // TODO 1: On each Neumann face, add the face quadrature for
      // int_F g_N * phi_i to cell_rhs(i) with FEFaceValues. Do this
      // before scattering the local system through the constraints.

      cell->get_dof_indices(indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, indices, system_matrix, system_rhs, true);
    }
}

void
Poisson::solve()
{
  SolverControl            control(4000, 1e-12);
  SolverCG<Vector<double>> solver(control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  constraints.distribute(solution);
}

void
Poisson::estimate()
{
  QGauss<2> quadrature(fe.degree + 3);

  FEValues<2>    values(fe,
                     quadrature,
                     update_JxW_values | update_quadrature_points);
  Vector<double> squared_indicators(triangulation.n_active_cells());
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      const auto index = cell->active_cell_index();
      values.reinit(cell);
      double squared_norm = 0.;
      for (const auto q : values.quadrature_point_indices())
        {
          const double f = forcing.value(values.quadrature_point(q));
          squared_norm += f * f * values.JxW(q);
        }
      // On these axis-aligned squares, the cellwise Laplacian of Q1 is zero.
      squared_cell_norms[index] = squared_norm;
      const double h            = cell->diameter();
      squared_indicators[index] = h * h * squared_norm;
    }

  if (use_face_residuals)
    add_face_residuals(squared_indicators);

  for (unsigned int i = 0; i < estimator.size(); ++i)
    estimator[i] = std::sqrt(squared_indicators[i]);
}

void
Poisson::add_face_residuals(Vector<double> &squared_indicators) const
{
  using Cell          = DoFHandler<2>::active_cell_iterator;
  using Scratch       = MeshWorker::ScratchData<2>;
  using Contributions = std::vector<std::pair<unsigned int, double>>;

  const Scratch scratch(fe,
                        QGauss<2>(fe.degree + 3),
                        update_default,
                        QGauss<1>(fe.degree + 3),
                        update_gradients | update_quadrature_points |
                          update_normal_vectors | update_JxW_values);

  const auto boundary_worker = [&](const Cell    &cell,
                                   unsigned int   face,
                                   Scratch       &local,
                                   Contributions &out) {
    if (cell->face(face)->boundary_id() != neumann_id)
      return;

    const auto               &values = local.reinit(cell, face);
    std::vector<Tensor<1, 2>> gradients(values.n_quadrature_points);
    values.get_function_gradients(solution, gradients);
    double squared_norm = 0.;
    for (const auto q : values.quadrature_point_indices())
      {
        const double r = neumann_data.value(values.quadrature_point(q)) -
                         gradients[q] * values.normal_vector(q);
        squared_norm += r * r * values.JxW(q);
      }
    out.emplace_back(cell->active_cell_index(),
                     cell->face(face)->diameter() * squared_norm);
  };

  const auto face_worker = [&](const Cell    &cell,
                               unsigned int   face,
                               unsigned int   subface,
                               const Cell    &neighbor,
                               unsigned int   neighbor_face,
                               unsigned int   neighbor_subface,
                               Scratch       &local,
                               Contributions &out) {
    const auto &interface = local.reinit(
      cell, face, subface, neighbor, neighbor_face, neighbor_subface);

    // TODO 3: Use squared_normal_jump(interface, solution), weighted by
    // the diameter of the physical integration segment. Add half the
    // weighted value to each cell in out (as an index/value pair).
    (void)interface;
    (void)out;
  };

  const auto copier = [&](const Contributions &out) {
    for (const auto &[index, value] : out)
      squared_indicators[index] += value;
  };

  MeshWorker::mesh_loop(dof_handler.begin_active(),
                        dof_handler.end(),
                        {},
                        copier,
                        scratch,
                        Contributions{},
                        MeshWorker::assemble_boundary_faces |
                          MeshWorker::assemble_own_interior_faces_once,
                        boundary_worker,
                        face_worker);
}

bool
Poisson::mark()
{
  if (refinement == "uniform")
    {
      triangulation.refine_global(1);
      return true;
    }

  // Lab 7 marks with the unweighted cell norms. When face residuals are
  // enabled, the same rules act on the full local indicators.
  Vector<double> scores(squared_cell_norms);
  for (unsigned int i = 0; i < scores.size(); ++i)
    scores[i] = use_face_residuals ? estimator[i] * estimator[i] : scores[i];

  const double sum = scores.l1_norm();
  const double n   = scores.size();
  const double threshold =
    refinement == "mean-square" ? sum / n : sum / (n * n);
  unsigned int marked = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (scores[cell->active_cell_index()] > threshold)
      {
        cell->set_refine_flag();
        ++marked;
      }
  std::cout << "Marked cells: " << marked << '/' << scores.size() << '\n';
  if (marked > 0)
    triangulation.execute_coarsening_and_refinement();
  return marked > 0;
}

void
Poisson::output_solution(unsigned int cycle) const
{
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);

  data_out.add_data_vector(solution, "solution");
  data_out.add_data_vector(estimator, "indicator", DataOut<2>::type_cell_data);
  data_out.build_patches();

  std::string   fname = "solution" + std::to_string(cycle) + ".vtu";
  std::ofstream output_file(fname);
  data_out.write_vtu(output_file);

  std::ofstream                                      pvd_record("solution.pvd");
  static std::vector<std::pair<double, std::string>> times_and_names;
  times_and_names.push_back(std::make_pair(static_cast<double>(cycle), fname));

  DataOutBase::write_pvd_record(pvd_record, times_and_names);
}

void
Poisson::run()
{
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  if (mixed_boundary)
    for (const auto &face : triangulation.begin_active()->face_iterators())
      if (face->at_boundary() && face->center()[0] == 1.)
        face->set_boundary_id(neumann_id);
  triangulation.refine_global(initial_refinement);

  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      setup_system();
      assemble_system();
      solve();
      error_table.error_from_exact(dof_handler, solution, exact_solution);
      estimate();
      std::cout << "Cycle " << cycle << ", cells "
                << triangulation.n_active_cells()
                << ", eta = " << estimator.l2_norm() << '\n';
      output_solution(cycle);
      if (cycle + 1 < n_cycles && !mark())
        break;
    }

  error_table.output_table(std::cout);
}

void
inspect_interface(const bool refine_right)
{
  Triangulation<2> mesh;
  GridGenerator::subdivided_hyper_rectangle(mesh,
                                            {2, 1},
                                            Point<2>(0., 0.),
                                            Point<2>(2., 1.));
  if (refine_right)
    {
      (++mesh.begin_active())->set_refine_flag();
      mesh.execute_coarsening_and_refinement();
    }

  const FE_Q<2> fe(1);
  DoFHandler<2> dofs(mesh);
  dofs.distribute_dofs(fe);
  const FunctionParser<2> field("2*abs(x-1)");
  Vector<double>          coefficients(dofs.n_dofs());
  VectorTools::interpolate(dofs, field, coefficients);
  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dofs, constraints);
  constraints.close();
  constraints.distribute(coefficients);

  FEInterfaceValues<2> interface(fe,
                                 QGauss<1>(2),
                                 update_gradients | update_normal_vectors |
                                   update_quadrature_points |
                                   update_JxW_values);

  double total_length = 0.;
  double total_jump   = 0.;
  for (const auto &right : dofs.active_cell_iterators())
    {
      // On this mesh the cells touching x=1 from the right use local face 0.
      const unsigned int right_face = 0;
      if (right->face(right_face)->center()[0] != 1.)
        continue;

      const auto   left = right->neighbor(right_face);
      unsigned int left_face;
      unsigned int left_subface = whole_face;
      if (right->neighbor_is_coarser(right_face))
        {
          // Predict this pair from the mesh diagram before inspecting it.
          const auto indices = right->neighbor_of_coarser_neighbor(right_face);
          left_face          = indices.first;
          left_subface       = indices.second;
        }
      else
        left_face = right->neighbor_of_neighbor(right_face);

      interface.reinit(
        right, right_face, whole_face, left, left_face, left_subface);
      const auto &from_right = interface.get_fe_face_values(0);
      const auto &from_left  = interface.get_fe_face_values(1);
      double      length     = 0.;
      std::cout << "Right face " << right_face << ", left face " << left_face;
      if (left_subface != whole_face)
        std::cout << ", left subface " << left_subface;
      std::cout << '\n';
      for (const auto q : interface.quadrature_point_indices())
        {
          const auto &x_right = from_right.quadrature_point(q);
          const auto &x_left  = from_left.quadrature_point(q);
          AssertThrow(x_right.distance(x_left) < 1e-12,
                      ExcMessage("The two quadrature points differ."));
          std::cout << "  q=" << q << "   right (" << x_right << ")"
                    << "   left (" << x_left << ")\n";
          length += interface.JxW(q);
        }
      const double jump = squared_normal_jump(interface, coefficients);
      std::cout << "  length = " << length << ", integral of j^2 = " << jump
                << '\n';
      total_length += length;
      total_jump += jump;
    }
  std::cout << "Interface total: length = " << total_length
            << ", integral of j^2 = " << total_jump << '\n';
}

int
main(int argc, char **argv)
{
  if (argc != 2)
    {
      std::cerr
        << "Usage: lab-08 path/to/poisson.prm | --conforming | --subfaces\n";
      return 1;
    }

  try
    {
      const std::string argument = argv[1];
      if (argument == "--conforming" || argument == "--subfaces")
        inspect_interface(argument == "--subfaces");
      else
        {
          Poisson problem(argument);
          problem.run();
        }
    }
  catch (const std::exception &error)
    {
      std::cerr << error.what() << '\n';
      return 1;
    }
}
