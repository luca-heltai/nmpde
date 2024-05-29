# Lab: Solving the Poisson Equation in deal.II

This laboratory session is designed to help you understand how to solve the Poisson equation using the deal.II library. You will learn how to create and refine meshes, handle degrees of freedom, define finite elements, assemble and solve the system of equations, and visualize the results.

## Objectives

By the end of this laboratory, you should be able to:

1. Create and refine meshes using the `Triangulation` class.
2. Set up and distribute degrees of freedom using the `DoFHandler` class.
3. Define finite elements with the `FE_Q` class.
4. Assemble the system of equations for a Poisson problem.
5. Solve the system using an iterative solver.
6. Output solutions using the `DataOut` class.
7. Compute and validate the solution by comparing with the exact solution.

## Example Code

Here is the example code snippet provided to you:

```cpp
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

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
struct PoissonParameters
{
  PoissonParameters()
  {
    prm.enter_subsection("Poisson parameters");
    {
      prm.add_parameter("Finite element degree", fe_degree);
      prm.add_parameter("Initial refinement", initial_refinement);
      prm.add_parameter("Number of cycle", n_cycles);
      prm.add_parameter("Exact solution expression", exact_solution_expression);
      prm.add_parameter("Right hand side expression", rhs_expression);
    }
    prm.leave_subsection();

    try
      {
        prm.parse_input("poisson_" + std::to_string(dim) + "d.prm");
      }
    catch (std::exception &exc)
      {
        prm.print_parameters("poisson_" + std::to_string(dim) + "d.prm");
        prm.parse_input("poisson_" + std::to_string(dim) + "d.prm");
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
  std::string  exact_solution_expression = "cos(pi*x)*cos(pi*y)";
  std::string  rhs_expression            = "2*pi*pi*cos(pi*x)*cos(pi*y)";

  FunctionParser<dim> exact_solution;
  FunctionParser<dim> rhs_function;

  ParameterHandler prm;
};



template <int dim>
class Poisson
{
public:
  Poisson(const PoissonParameters<dim> &parameters);
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
  output_results() const;

  const PoissonParameters<dim> &par;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};



template <int dim>
Poisson<dim>::Poisson(const PoissonParameters<dim> &par)
  : par(par)
  , fe(par.fe_degree)
  , dof_handler(triangulation)
{}



template <int dim>
void
Poisson<dim>::make_grid()
{
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(par.initial_refinement);

  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;
}


template <int dim>
void
Poisson<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}



template <int dim>
void
Poisson<dim>::assemble_system()
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

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          {
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx

            const auto &x_q = fe_values.quadrature_point(q_index);
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            par.rhs_function.value(x_q) *       // f(x_q)
                            fe_values.JxW(q_index));            // dx
          }

      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
        {
          for (const unsigned int j : fe_values.dof_indices())
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           par.exact_solution,
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
}



template <int dim>
void
Poisson<dim>::solve()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;
}



template <int dim>
void
Poisson<dim>::output_results() const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");

  data_out.build_patches();

  std::ofstream output(dim == 2 ? "solution-2d.vtu" : "solution-3d.vtu");
  data_out.write_vtu(output);
}



template <int dim>
void
Poisson<dim>::run()
{
  std::cout << "Solving problem in " << dim << " space dimensions."
            << std::endl;

  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}



int
main()
{
  {
    PoissonParameters<2> par;
    Poisson<2>           laplace_problem_2d(par);
    laplace_problem_2d.run();
  }

  {
    PoissonParameters<3> par;
    Poisson<3>           laplace_problem_3d(par);
    laplace_problem_3d.run();
  }

  return 0

;
}
```

## Exercises

### Exercise 1: Understanding the PoissonParameters Class

1. **Parameter File:**
   - Create a parameter file named `poisson_2d.prm` and `poisson_3d.prm` with the following contents:

     ```
     subsection Poisson parameters
       set Finite element degree = 1
       set Initial refinement = 3
       set Number of cycle = 1
       set Exact solution expression = cos(pi*x)*cos(pi*y)
       set Right hand side expression = 2*pi*pi*cos(pi*x)*cos(pi*y)
     end
     ```

   - Ensure these files are in the same directory as your executable.

2. **Modify Parameters:**
   - Change the `Finite element degree` and `Initial refinement` parameters in the parameter files and observe how they affect the solution.

### Exercise 2: Creating and Refining Meshes

1. **Mesh Creation:**
   - Modify the `make_grid` function to create different types of meshes using `GridGenerator`, such as `hyper_ball` and `subdivided_hyper_rectangle`.
   - Print the number of active cells and total cells for each mesh.
   - Use `GridGenerator::generate_from_name_and_arguments`, and add two paramaters to the parameter file to generate the grid from the function name (i.e., `hyper_cube`, or `hyper_shell`) and the function arguments

### Exercise 3: Assembling and Solving the System

1. **Understanding Assembly:**
   - Add comments to the `assemble_system` function to explain each step in the assembly process.
   - Modify the assembly process to use different quadrature formulas and observe how this affects the solution accuracy.

2. **Solving the System:**
   - Experiment with different solvers and preconditioners available in deal.II, such as `SolverGMRES` and `PreconditionJacobi`.
   - Compare the number of iterations needed for convergence and the accuracy of the solution.

### Exercise 4: Output and Visualization

1. **Output Solutions:**
   - Output the solution to different formats, such as `VTU` and `VTK`.
   - Use Paraview to visualize the solutions and create contour plots.

2. **Higher-Order Output:**
   - Modify the `output_results` function to output higher-order elements by adjusting the `DataOut` settings.
   - Visualize the higher-order solutions and compare them with the linear solutions.

### Exercise 5: Error Analysis

1. **Compute Errors:**
   - Compute the $L^2$ and $H^1$ errors of the solution using `VectorTools::integrate_difference`.
   - Compare these errors with the analytical errors obtained from the exact solution.

2. **Error Convergence:**
   - Refine the mesh globally multiple times and compute the errors for each refinement level.
   - Plot the errors as a function of the number of degrees of freedom and analyze the convergence rates.
