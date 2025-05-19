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
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

template <int dim>
struct StokesParameters
{
  StokesParameters()
    : exact_solution(dim + 1)
    , rhs_function(dim + 1)
    , neumann_function(dim + 1)
    , convergence_table(dim == 2 ?
                          std::vector<std::string>({{"u", "u", "p"}}) :
                          std::vector<std::string>({{"u", "u", "u", "p"}}),
                        {{VectorTools::H1_norm, VectorTools::L2_norm},
                         {VectorTools::L2_norm}})
  {
    prm.enter_subsection("Stokes parameters");
    {
      prm.add_parameter("Finite element degree", fe_degree);
      prm.add_parameter("Initial refinement", initial_refinement);
      prm.add_parameter("Number of cycles", n_cycles);
      prm.add_parameter("Exact solution expression", exact_solution_expression);
      prm.add_parameter("Right hand side expression", rhs_expression);
      prm.add_parameter("Neumann boundary expression", neumann_expression);
      prm.add_parameter("Neumann boundary ids", neumann_boundary_ids);
      prm.add_parameter("Viscosity", eta);
    }
    prm.leave_subsection();

    prm.enter_subsection("Convergence table");
    convergence_table.add_parameters(prm);
    prm.leave_subsection();

    try
      {
        prm.parse_input("stokes_" + std::to_string(dim) + "d.prm");
      }
    catch (std::exception &exc)
      {
        prm.print_parameters("stokes_" + std::to_string(dim) + "d.prm");
        prm.parse_input("stokes_" + std::to_string(dim) + "d.prm");
      }
    std::map<std::string, double> constants;
    constants["pi"] = numbers::PI;
    exact_solution.initialize(FunctionParser<dim>::default_variable_names(),
                              {exact_solution_expression},
                              constants);
    rhs_function.initialize(FunctionParser<dim>::default_variable_names(),
                            {rhs_expression},
                            constants);
    neumann_function.initialize(FunctionParser<dim>::default_variable_names(),
                                {neumann_expression},
                                constants);
  }
  unsigned int fe_degree          = 1;
  unsigned int initial_refinement = 3;
  unsigned int n_cycles           = 1;

  double eta = 1.0;

  std::string                  exact_solution_expression = "0; 0; 0";
  std::string                  rhs_expression            = "0; -1; 0";
  std::string                  neumann_expression        = "0; 0; 0";
  std::set<types::boundary_id> neumann_boundary_ids      = {};

  FunctionParser<dim> exact_solution;
  FunctionParser<dim> rhs_function;
  FunctionParser<dim> neumann_function;

  mutable ParsedConvergenceTable convergence_table;

  ParameterHandler prm;
};



template <int dim>
class Stokes
{
public:
  Stokes(const StokesParameters<dim> &parameters);
  void
  run();

private:
  void
  make_grid();
  void
  estimate();
  void
  mark();
  void
  refine();
  void
  setup_system();
  void
  assemble_system();
  void
  solve();
  void
  output_results(const unsigned int cycle) const;

  const StokesParameters<dim> &par;

  Triangulation<dim> triangulation;
  FESystem<dim>      fe;
  DoFHandler<dim>    dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;

  Vector<float> estimated_error_per_cell;

  FEValuesExtractors::Vector velocity;
  FEValuesExtractors::Scalar pressure;
};



template <int dim>
Stokes<dim>::Stokes(const StokesParameters<dim> &par)
  : par(par)
  , fe(FE_Q<dim>(par.fe_degree), dim, FE_Q<dim>(par.fe_degree), 1)
  , dof_handler(triangulation)
  , velocity(0)
  , pressure(dim)
{}



template <int dim>
void
Stokes<dim>::make_grid()
{
  GridGenerator::hyper_cube(triangulation, -1, 1, true);
  triangulation.refine_global(par.initial_refinement);

  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;
}



template <int dim>
void
Stokes<dim>::estimate()
{
  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim - 1>(fe.degree + 1),
                                     {},
                                     solution,
                                     estimated_error_per_cell);
}

template <int dim>
void
Stokes<dim>::mark()
{
  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                  estimated_error_per_cell,
                                                  0.3,
                                                  0.03);
}

template <int dim>
void
Stokes<dim>::refine()
{
  triangulation.execute_coarsening_and_refinement();
}



template <int dim>
void
Stokes<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  constraints.clear();
  auto all_boundary_ids = triangulation.get_boundary_ids();
  std::set<types::boundary_id> dirichlet_boundary_ids;
  for (const auto &id : all_boundary_ids)
    if (par.neumann_boundary_ids.find(id) == par.neumann_boundary_ids.end())
      dirichlet_boundary_ids.insert(id);

  for (const auto &id : dirichlet_boundary_ids)
    VectorTools::interpolate_boundary_values(dof_handler,
                                             id,
                                             par.exact_solution,
                                             constraints);

  // Create hanging node constraints
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  estimated_error_per_cell.reinit(triangulation.n_active_cells());
}



template <int dim>
void
Stokes<dim>::assemble_system()
{
  QGauss<dim>     quadrature_formula(fe.degree + 1);
  QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(fe,
                                   face_quadrature_formula,
                                   update_values | update_quadrature_points |
                                     update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          {
            const unsigned int comp_i = fe.system_to_component_index(i).first;

            const auto v_i      = fe_values[velocity].value(i, q_index);
            const auto div_v_i  = fe_values[velocity].divergence(i, q_index);
            const auto grad_v_i = fe_values[velocity].gradient(i, q_index);

            const auto q_i = fe_values[pressure].value(i, q_index);

            for (const unsigned int j : fe_values.dof_indices())
              {
                const auto grad_u_j = fe_values[velocity].gradient(j, q_index);
                const auto div_u_j = fe_values[velocity].divergence(j, q_index);
                const auto p_j     = fe_values[pressure].value(j, q_index);

                cell_matrix(i, j) +=
                  (par.eta * scalar_product(grad_u_j, grad_v_i) +
                   div_v_i * p_j + div_u_j * q_i) *
                  fe_values.JxW(q_index); // dx
              }

            const auto &x_q = fe_values.quadrature_point(q_index);
            if (comp_i < dim)
              cell_rhs(i) += (v_i[comp_i] * // phi_i(x_q)
                              par.rhs_function.value(x_q, comp_i) * // f(x_q)
                              fe_values.JxW(q_index));              // dx
          }

      // Neumann boundary condition
      for (const auto &f : cell->face_indices())
        if (cell->face(f)->at_boundary() &&
            par.neumann_boundary_ids.find(cell->face(f)->boundary_id()) !=
              par.neumann_boundary_ids.end())
          {
            fe_face_values.reinit(cell, f);
            for (const unsigned int q_index :
                 fe_face_values.quadrature_point_indices())
              for (const unsigned int i : fe_face_values.dof_indices())
                cell_rhs(i) +=
                  (fe_face_values.shape_value(i, q_index) * // phi_i(x_q)
                   par.neumann_function.value(
                     fe_face_values.quadrature_point(q_index)) * // g(x_q)
                   fe_face_values.JxW(q_index));                 // ds
          }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      // for (const unsigned int i : fe_values.dof_indices())
      //   {
      //     for (const unsigned int j : fe_values.dof_indices())
      //       system_matrix.add(local_dof_indices[i],
      //                         local_dof_indices[j],
      //                         cell_matrix(i, j));

      //     system_rhs(local_dof_indices[i]) += cell_rhs(i);
      //   }
    }

  // std::map<types::global_dof_index, double> boundary_values;
  // VectorTools::interpolate_boundary_values(dof_handler,
  //                                          0,
  //                                          par.exact_solution,
  //                                          boundary_values);
  // MatrixTools::apply_boundary_values(boundary_values,
  //                                    system_matrix,
  //                                    solution,
  //                                    system_rhs);
}



template <int dim>
void
Stokes<dim>::solve()
{
  SparseDirectUMFPACK solver;
  solver.initialize(system_matrix);
  solver.vmult(solution, system_rhs);
  constraints.distribute(solution);
}



template <int dim>
void
Stokes<dim>::output_results(const unsigned int cycle) const
{
  DataOut<dim> data_out;

  std::vector<std::string> names(dim + 1, "u");
  names[dim] = "p"; // last component is pressure

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);


  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution,
                           names,
                           DataOut<dim>::type_dof_data,
                           component_interpretation);
  data_out.add_data_vector(estimated_error_per_cell, "estimator");

  data_out.build_patches();

  auto fname =
    "solution-" + std::to_string(dim) + "d_" + std::to_string(cycle) + ".vtu";

  std::ofstream output(fname);
  data_out.write_vtu(output);

  static std::vector<std::pair<double, std::string>> times_and_names;
  times_and_names.push_back({cycle, fname});

  std::ofstream pvd_output("solution-" + std::to_string(dim) + "d.pvd");

  DataOutBase::write_pvd_record(pvd_output, times_and_names);
}



template <int dim>
void
Stokes<dim>::run()
{
  std::cout << "Solving problem in " << dim << " space dimensions."
            << std::endl;

  for (unsigned int cycle = 0; cycle < par.n_cycles; ++cycle)
    {
      if (cycle == 0)
        make_grid();
      else
        {
          mark();
          refine();
        }
      setup_system();
      assemble_system();
      solve();
      estimate();
      output_results(cycle);
      par.convergence_table.error_from_exact(dof_handler,
                                             solution,
                                             par.exact_solution);
    }
  par.convergence_table.output_table(std::cout);
}



int
main()
{
  {
    StokesParameters<2> par;
    Stokes<2>           laplace_problem_2d(par);
    laplace_problem_2d.run();
  }

  // {
  //   StokesParameters<3> par;
  //   Stokes<3>           laplace_problem_3d(par);
  //   laplace_problem_3d.run();
  // }

  return 0;
}
