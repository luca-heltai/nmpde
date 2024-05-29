# Lab 5: Improving the Poisson Solver (ParsedConvergenceTable and AffineConstraints)

This laboratory session is designed to help you understand how to improve the Poisson solver to analyze the convergence of your numerical solutions, and enable local refinement. You will learn how to create locally refined meshes, handle degrees of freedom with hanging nodes, and compute convergence rates.

## Objectives

By the end of this laboratory, you should be able to:

1. Create locally refined meshes, and use hanging node constraintgs
2. Replace `MatrixTools::apply_boundary_values` with `AffineConstraints`
3. Analyze convergence rates using the `ParsedConvergenceTable` class.

## Example Code

Here is the example code snippet provided to you:

```cpp
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
struct PoissonParameters
{
  PoissonParameters()
  {
    prm.enter_subsection("Poisson parameters");
    {
      prm.add_parameter("Finite element degree", fe_degree);
      prm.add_parameter("Initial refinement", initial_refinement);
      prm.add_parameter("Number of cycles", n_cycles);
      prm.add_parameter("Exact solution expression", exact_solution_expression);
      prm.add_parameter("Right hand side expression", rhs_expression);
    }
    prm.leave_subsection();

    prm.enter_subsection("Convergence table");
    convergence_table.add_parameters(prm);
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

  mutable ParsedConvergenceTable convergence_table;

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
  output_results(const unsigned int cycle) const;

  const PoissonParameters<dim> &par;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  AffineConstraints<double> constraints;

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

  constraints.clear();
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
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
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}



template <int dim>
void
Poisson<dim>::solve()
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
Poisson<dim>::output_results(const unsigned int cycle) const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");

  data_out.build_patches();

  auto fname =
    "solution-" + std::to_string(dim) + "d_" + std::to_string(cycle) + ".vtu";

  std::ofstream output(fname);
  data_out.write_vtu(output);

  static std::vector<std::pair<double, std::string>> times_and_names;
  times_and_names.push_back({cycle, fname});

  std::ofstream pvd_output("solution-" + std::to_string(dim) + "d.pvd");

  DataOutBase::write_pvd_record(pvd_output,

 times_and_names);
}



template <int dim>
void
Poisson<dim>::run()
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
    PoissonParameters<2> par;
    Poisson<2>           laplace_problem_2d(par);
    laplace_problem_2d.run();
  }

  return 0;
}
```

## Using `ParsedConvergenceTable` for Convergence Analysis

The `ParsedConvergenceTable` class in deal.II simplifies the creation and management of convergence tables. This class reads options for generating convergence tables from a parameter file and provides methods to compute errors using reference exact solutions, differences between numerical solutions, or custom error computations via `std::function` objects.

### Overview

Here is a brief overview of the main features and usage of `ParsedConvergenceTable`:

1. **Adding Parameters:**
   - The class adds necessary parameters to a `ParameterHandler` object, allowing configuration via a parameter file.

2. **Computing Errors:**
   - Methods like `error_from_exact()` and `difference()` can be used to compute various norms of the error.

3. **Outputting the Table:**
   - The final convergence table can be outputted to a stream or a file.

### Example Usage

The following example demonstrates a typical usage of `ParsedConvergenceTable`:

```cpp
ParsedConvergenceTable table;
ParameterHandler prm;
table.add_parameters(prm);

for (unsigned int i = 0; i < n_cycles; ++i)
{
  // ... perform computations for the i-th cycle
  table.error_from_exact(dof_handler, solution, exact_solution);
}
table.output_table(std::cout);
```

In this example:

- A `ParsedConvergenceTable` object is created.
- Parameters are added to a `ParameterHandler`.
- Errors are computed for each cycle using `error_from_exact()`.
- The convergence table is outputted to the console.

### Features

#### Parameter Configuration

By calling `add_parameters()` and passing a `ParameterHandler` object, several options are defined, which can be modified at runtime through a parameter file:

```plaintext
set Enable computation of the errors = true
set Error file name                  =
set Error precision                  = 3
set Exponent for p-norms             = 2
set Extra columns                    = dofs, cells
set List of error norms to compute   = Linfty_norm, L2_norm, H1_norm
set Rate key                         = dofs
set Rate mode                        = reduction_rate_log2
```

These parameters control the computation and output of the convergence table.

#### Error Computation

Whenever `error_from_exact()` or `difference()` is called, the class:

- Inspects its parameters.
- Computes all specified norms.
- Computes any extra columns defined via `add_extra_column()`.
- Writes one row of the convergence table.

#### Output

The convergence table can be outputted using `output_table()`:

- To a stream (e.g., `std::cout`).
- To a file specified in the parameter file.

## Using AffineConstraints for Boundary Conditions and Hanging Nodes

In deal.II, boundary conditions and constraints for hanging nodes can be handled more flexibly using `AffineConstraints<double>` instead of `MatrixTools::apply_boundary_values`. The `AffineConstraints` class allows you to manage multiple types of constraints in a unified way.

### Setting Up AffineConstraints

To replace `MatrixTools::apply_boundary_values` with `AffineConstraints<double>`, follow these steps:

1. **Clear and Initialize Constraints:**
   Clear any existing constraints and set up the boundary values and hanging nodes constraints using `AffineConstraints<double>`.

2. **Distribute Local to Global:**
   Use the `distribute_local_to_global` method of `AffineConstraints<double>` during the assembly of the system matrix and right-hand side vector.

3. **Distribute Solution:**
   After solving the linear system, use `AffineConstraints<double>` to apply the constraints to the solution vector.

### Comparison with MatrixTools::apply_boundary_values

Using `AffineConstraints` has several advantages over `MatrixTools::apply_boundary_values`:

- **Flexibility:** `AffineConstraints` can handle multiple types of constraints (e.g., boundary values, hanging nodes) in a unified manner.
- **Simplicity:** The constraints are applied automatically during the assembly and solution phases, reducing the need for additional function calls.
- **Efficiency:** The constraints are incorporated directly into the system matrix and right-hand side vector, during the assembly time, eliminating the cost of applying algebraic constraints as a post-processing steps of linear algebra objects.

## Exercises

### Exercise 1: Understanding the PoissonParameters Class

1. **Parameter File:**
   - Create a parameter file named `poisson_2d.prm` with the following contents:

     ```plaintext
     subsection Poisson parameters
       set Finite element degree = 1
       set Initial refinement = 3
       set Number of cycles = 5
       set Exact solution expression = cos(pi*x)*cos(pi*y)
       set Right hand side expression = 2*pi*pi*cos(pi*x)*cos(pi*y)
     end

     subsection Convergence table
       set Enable computation of the errors = true
       set Error file name                  = errors.txt
       set Error precision                  = 3
       set Exponent for p-norms             = 2
       set Extra columns                    = dofs, cells
       set List of error norms to compute   = Linfty_norm, L2_norm, H1_norm
       set Rate key                         = dofs
       set Rate mode                        = reduction_rate_log2
     end
     ```

   - Ensure this file is in the same directory as your executable.

2. **Modify Parameters:**
   - Change the `Finite element degree` and `Initial refinement` parameters in the parameter file and observe how they affect the solution and convergence.

### Exercise 2: Refining Meshes

1. **Local Refinement:**
   - Modify the parameter file to add parameters controlling local refinement:

     ```plaintext
     subsection Local refinement
       set Refinement fraction = 0.3
       set Coarsening fraction = 0.0
       set Minimum grid level = 3
       set Maximum grid level = 7
     end
     ```

   - Implement local mesh refinement using Kelly error estimator, replace the current placeholder for mesh refinement with

     ```cpp
     for (unsigned int cycle = 0; cycle < par.n_cycles; ++cycle)
     {
       if (cycle == 0)
         make_grid();
       else
       {
         Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
         KellyErrorEstimator<dim>::estimate(dof_handler,
                                            QGauss<dim-1>(fe.degree + 1),
                                            {},
                                            solution,
                                            estimated_error_per_cell);
         
         GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                           estimated_error_per_cell,
                                                           0.3, // Refinement fraction from parameter file
                                                           0.0); // Coarsening fraction from parameter file
         
         triangulation.execute_coarsening_and_refinement();
       }
       ...
     ```

### Exercise 3: Convergence Rate Analysis - Local Refinement vs. Global Refinement

1. Create two parameter files, one for global refinement (`global_refinement.prm`, i.e., setting refinement fraction to `1.0`) and one for local refinement (`local_refinement.prm`), choosing the coefficients such that the total number of degrees of freedom in the final step is the same for the two cases (i.e., more cycles in the local refinement case, tuning the top fraction parameter)

2. Analyse the two error tables. What can you say about the final error in the two cases? Which one works better?

3. Analyse the convergence rate in the two cases w.r.t. the number of degrees of freedom. What powers do you see for the error decay in the two cases, in L2 and H1?

4. Do the same for degree 2. What powers do you see now?
