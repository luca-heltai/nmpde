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
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

template <int dim>
struct LinearElasticityParameters
{
  LinearElasticityParameters()
    : exact_solution(dim)
    , rhs_function(dim)
    , convergence_table({"u", "u"})
  {
    prm.enter_subsection("LinearElasticity parameters");
    {
      prm.add_parameter("Finite element degree", fe_degree);
      prm.add_parameter("Initial refinement", initial_refinement);
      prm.add_parameter("Number of cycles", n_cycles);
      prm.add_parameter("Exact solution expression", exact_solution_expression);
      prm.add_parameter("Right hand side expression", rhs_expression);
      prm.add_parameter("Lame coefficient mu", mu);
      prm.add_parameter("Lame coefficient lambda", lambda);
    }
    prm.leave_subsection();

    prm.enter_subsection("Convergence table");
    convergence_table.add_parameters(prm);
    prm.leave_subsection();

    try
      {
        prm.parse_input("LinearElasticity_" + std::to_string(dim) + "d.prm");
      }
    catch (std::exception &exc)
      {
        prm.print_parameters("LinearElasticity_" + std::to_string(dim) +
                             "d.prm");
        prm.parse_input("LinearElasticity_" + std::to_string(dim) + "d.prm");
      }
    std::map<std::string, double> constants;
    constants["pi"] = numbers::PI;
    exact_solution.initialize(FunctionParser<dim>::default_variable_names(),
                              {exact_solution_expression},
                              constants);
    rhs_function.initialize(FunctionParser<dim>::default_variable_names(),
                            {rhs_expression},
                            constants);
  }
  unsigned int fe_degree                 = 1;
  unsigned int initial_refinement        = 3;
  unsigned int n_cycles                  = 1;
  std::string  exact_solution_expression = "0; 0";
  std::string  rhs_expression            = "0; 0";
  double       mu                        = 1.0;
  double       lambda                    = 1.0;

  FunctionParser<dim> exact_solution;
  FunctionParser<dim> rhs_function;

  mutable ParsedConvergenceTable convergence_table;

  ParameterHandler prm;
};



template <int dim>
class LinearElasticity
{
public:
  LinearElasticity(const LinearElasticityParameters<dim> &parameters);
  void
  run();

private:
  void
  make_grid();
  void
  setup_system();
  void
  assemble_system();
  void
  solve();
  void
  output_results(const unsigned int cycle) const;

  const LinearElasticityParameters<dim> &par;

  Triangulation<dim> triangulation;
  FESystem<dim>      fe;
  DoFHandler<dim>    dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};



template <int dim>
LinearElasticity<dim>::LinearElasticity(
  const LinearElasticityParameters<dim> &par)
  : par(par)
  , fe(FE_Q<dim>(par.fe_degree), dim)
  , dof_handler(triangulation)
{}



template <int dim>
void
LinearElasticity<dim>::make_grid()
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
LinearElasticity<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  constraints.clear();
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           par.exact_solution,
                                           constraints);

  VectorTools::interpolate_boundary_values(dof_handler,
                                           1,
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
}



template <int dim>
void
LinearElasticity<dim>::assemble_system()
{
  QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  FEValuesExtractors::Vector displacements(0);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          {
            const auto &phi_i = fe_values[displacements].value(i, q_index);
            const auto &div_phi_i =
              fe_values[displacements].divergence(i, q_index);
            const auto &eps_phi_i =
              fe_values[displacements].symmetric_gradient(i, q_index);

            for (const unsigned int j : fe_values.dof_indices())
              {
                const auto &div_phi_j =
                  fe_values[displacements].divergence(j, q_index);
                const auto &eps_phi_j =
                  fe_values[displacements].symmetric_gradient(j, q_index);

                cell_matrix(i, j) +=
                  (par.mu * scalar_product(eps_phi_i, eps_phi_j) +
                   par.lambda * div_phi_i * div_phi_j) *
                  fe_values.JxW(q_index); // dx
              }

            const auto &x_q    = fe_values.quadrature_point(q_index);
            const auto  comp_i = fe.system_to_component_index(i).first;

            cell_rhs(i) += (phi_i[comp_i] *                       // phi_i(x_q)
                            par.rhs_function.value(x_q, comp_i) * // f(x_q)
                            fe_values.JxW(q_index));              // dx
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
LinearElasticity<dim>::solve()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  constraints.distribute(solution);

  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;
}



template <int dim>
void
LinearElasticity<dim>::output_results(const unsigned int cycle) const
{
  DataOut<dim>             data_out;
  std::vector<std::string> names(dim, "displacement");
  const std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution,
                           names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);

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
LinearElasticity<dim>::run()
{
  std::cout << "Solving problem in " << dim << " space dimensions."
            << std::endl;

  for (unsigned int cycle = 0; cycle < par.n_cycles; ++cycle)
    {
      if (cycle == 0)
        make_grid();
      else
        {
          // estimate error
          // refine mesh where error is larger
          for (const auto &cell : triangulation.active_cell_iterators())
            {
              if (cell->center().distance(Point<dim>(0.5, 0.5)) < 0.25)
                cell->set_refine_flag();
            }
          triangulation.execute_coarsening_and_refinement();
        }
      setup_system();
      assemble_system();
      solve();
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
    LinearElasticityParameters<2> par;
    LinearElasticity<2>           laplace_problem_2d(par);
    laplace_problem_2d.run();
  }

  // {
  //   LinearElasticityParameters<3> par;
  //   LinearElasticity<3>           laplace_problem_3d(par);
  //   laplace_problem_3d.run();
  // }

  return 0;
}
