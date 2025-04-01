#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
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

/**
 * @brief Stokes equations for a viscous fluid
 *
 * The Stokes equations describe the motion of a viscous, incompressible fluid.
 *
 * @f[
 * \nabla p + \rho_f (\mathbf{u} \cdot \nabla) \mathbf{u} - \mu_f \nabla^2
 * \mathbf{u} = 0 \nabla \cdot \mathbf{u} = 0
 * @f]
 *
 * Where:
 * - @f$ p @f$ is the fluid pressure
 * - @f$ \rho_f @f$ is the fluid density
 * - @f$ \mathbf{u} @f$ is the fluid velocity vector
 * - @f$ \mu_f @f$ is the fluid dynamic viscosity
 *
 * The first equation (1) represents the conservation of momentum, while the
 * second equation represents the incompressibility constraint.
 */
template <int dim>
class Stokes
{
public:
  /**
   * @fn Stokes(const StokesParameters<dim> &parameters)
   * @brief Constructor that takes a StokesParameters object as argument.
   *
   * Initializes the class with the given parameters.
   */
  Stokes(const StokesParameters<dim> &parameters);

  /**
   * @fn void run()
   * @brief Method to run the simulation.
   *
   * This method sets up and solves the Stokes equations.
   */
  void
  run();

private:
  /**
   * @fn void make_grid()
   * @brief Private method to create the grid.
   *
   * Creates a mesh for the computational domain.
   */
  void
  make_grid();

  /**
   * @fn void setup_system()
   * @brief Private method to set up the system.
   *
   * Initializes the finite element system and sets up the matrix structure.
   */
  void
  setup_system();

  /**
   * @fn void assemble_system()
   * @brief Private method to assemble the system matrix.
   *
   * Assembles the linear system matrix from the contributions of each finite
   * element.
   */
  void
  assemble_system();

  /**
   * @fn void solve()
   * @brief Private method to solve the linear system.
   *
   * Solves the linear system using a chosen solver.
   */
  void
  solve();

  /**
   * @fn void output_results(const unsigned int cycle) const


template <int dim>
Stokes<dim>::Stokes(const StokesParameters<dim> &par)
  : par(par)
  , fe(FE_Q<dim>(par.fe_degree), dim, FE_DGP<dim>(par.fe_degree - 1), 1)
  , dof_handler(triangulation)
{}



template <int dim>
void
Stokes<dim>::make_grid()
{
  // Generate a hyper shell

  


  GridGenerator::channel_with_cylinder(triangulation,
                                       par.domain_shell_region_width,
                                       par.domain_n_shells,
                                       par.domain_skewness,
                                       true);

  triangulation.refine_global(par.initial_refinement);

  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;
}


template <int dim>
void
Stokes<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  constraints.clear();
  for (const auto &id : par.dirichlet_ids)
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

  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          {
            const auto &v_i        = fe_values[velocity].value(i, q_index);
            const auto &div_v_i    = fe_values[velocity].divergence(i, q_index);
            const auto &grad_phi_i = fe_values[velocity].gradient(i, q_index);
            const auto &q_i        = fe_values[pressure].value(i, q_index);

            for (const unsigned int j : fe_values.dof_indices())
              {
                // const auto &v_j = fe_values[velocity].value(j, q_index);
                const auto &div_v_j =
                  fe_values[velocity].divergence(j, q_index);
                const auto &grad_phi_j =
                  fe_values[velocity].gradient(j, q_index);
                const auto &q_j = fe_values[pressure].value(j, q_index);


                cell_matrix(i, j) +=
                  (par.eta * scalar_product(grad_phi_i, grad_phi_j) +
                   div_v_i * q_j + q_i * div_v_j) *
                  fe_values.JxW(q_index); // dx
              }

            const auto &x_q    = fe_values.quadrature_point(q_index);
            const auto  comp_i = fe.system_to_component_index(i).first;

            if (comp_i < dim)
              cell_rhs(i) += (v_i[comp_i] * // phi_i(x_q)
                              par.rhs_function.value(x_q, comp_i) * // f(x_q)
                              fe_values.JxW(q_index));              // dx
          }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
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
  DataOut<dim>             data_out;
  std::vector<std::string> names(dim, "velocity");
  names.push_back("pressure");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);

  data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);

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
          // Estimate error
          Vector<float> estimated_error_per_cell(
            triangulation.n_active_cells());
          KellyErrorEstimator<dim>::estimate(dof_handler,
                                             QGauss<dim - 1>(fe.degree + 1),
                                             {},
                                             solution,
                                             estimated_error_per_cell);
          // Mark for refinement
          GridRefinement::refine_and_coarsen_fixed_number(
            triangulation,
            estimated_error_per_cell,
            par.top_fraction,
            par.bottom_fraction);

          // Actually refine
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
