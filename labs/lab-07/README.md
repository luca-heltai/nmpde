# Lab: Vector-Valued Finite Element Spaces in Linear Elasticity

## Overview

In this lab, you will work with vector-valued finite element spaces to solve linear elasticity problems using the deal.II library. You will learn about using `FESystem` for creating vector-valued finite elements and `FEValuesExtractors` for accessing individual components of the vector fields. Additionally, you will implement non-homogeneous Neumann boundary conditions and perform three different types of experiments: Dirichlet pulling experiment, Neumann pulling experiment, and cantilever experiment.

## Salient Changes for Vector-Valued Elements

### FESystem

`FESystem` is used to create a vector-valued finite element space by combining scalar finite elements. In this lab, `FESystem<dim>` is created using `FE_Q<dim>` elements:

```cpp
FESystem<dim> fe(FE_Q<dim>(par.fe_degree), dim);
```

### FEValuesExtractors

`FEValuesExtractors` are used to access specific components of the vector field in `FEValues` and `FEFaceValues` objects. For example, to extract displacement components:

```cpp
FEValuesExtractors::Vector displacements(0);
```

This extractor can then be used to access the values, gradients, and other quantities of the displacement field.

### Assembly and Boundary Conditions

In vector-valued problems, the assembly of the system matrix and right-hand side vector involves operations on vector fields. The implementation of boundary conditions (both Dirichlet and Neumann) also needs to be adapted for vector-valued elements.

#### Dirichlet Boundary Conditions

Dirichlet boundary conditions are imposed using `AffineConstraints<double>`:

```cpp
for (const auto &id : par.dirichlet_ids)
  VectorTools::interpolate_boundary_values(dof_handler,
                                           id,
                                           par.exact_solution,
                                           constraints);
```

#### Neumann Boundary Conditions

Neumann boundary conditions are implemented by integrating the Neumann data over the boundary faces:

```cpp
for (const auto &f : cell->face_indices())
  if (cell->face(f)->at_boundary() &&
      par.neumann_ids.find(cell->face(f)->boundary_id()) != par.neumann_ids.end())
    {
      fe_face_values.reinit(cell, f);
      for (const unsigned int q_index : fe_face_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          {
            const auto &phi_i = fe_face_values[displacements].value(i, q_index);
            const auto &x_q = fe_face_values.quadrature_point(q_index);
            const auto comp_i = fe.system_to_component_index(i).first;

            cell_rhs(i) += phi_i[comp_i] * par.neumann_function.value(x_q, comp_i) *
                           fe_face_values.JxW(q_index);
          }
    }
```

## Exercises

### Exercise 1: Implement Non-Homogeneous Neumann Boundary Conditions

1. **Modify the `assemble_system` Method:**
   - Ensure that non-homogeneous Neumann boundary conditions are correctly implemented as shown in the provided code.

### Exercise 2: Perform Dirichlet Pulling Experiment

1. **Setup:**
   - Fix one side of the domain (e.g., `x=0`) with Dirichlet boundary conditions.
   - Apply a uniform displacement on the opposite side (e.g., `x=1`).

2. **Run and Analyze:**
   - Run the simulation and visualize the displacement field.
   - Analyze the results and check if the deformation is as expected.

3. **Physical considerations**
   - What happens if you set `lambda` to 0 in this case?
   - What happens if you set `lambda` to `1e5` in this case?

### Exercise 3: Perform Neumann Pulling Experiment

1. **Setup:**
   - Fix one side of the domain (e.g., `x=0`) with Dirichlet boundary conditions.
   - Apply a uniform traction (Neumann boundary condition) on the opposite side (e.g., `x=1`).

2. **Run and Analyze:**
   - Run the simulation and visualize the displacement field.
   - Analyze the results and check if the deformation is as expected.

### Exercise 4: Perform Cantilever Experiment

1. **Setup:**
   - Fix one side of the domain (e.g., `x=0`) with Dirichlet boundary conditions.
   - Apply a point load or distributed load (Neumann boundary condition) at the free end (e.g., `x=1`).

2. **Run and Analyze:**
   - Run the simulation and visualize the displacement field.
   - Analyze the results and check if the deformation is as expected.

## Code Snippets

### Assembly with Non-Homogeneous Neumann Boundary Conditions

```cpp
template <int dim>
void LinearElasticity<dim>::assemble_system()
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

      for (const auto &f : cell->face_indices())
        if (cell->face(f)->at_boundary() &&
            par.neumann_ids.find(cell->face(f)->boundary_id()) !=
              par.neumann_ids.end())
          {
            fe_face_values.reinit(cell, f);
            for (const unsigned int q_index :
                 fe_face_values.quadrature_point_indices())
              for (const unsigned int i : fe_values.dof_indices())
                {
                  const auto &phi_i =
                    fe_face_values[displacements].value(i, q_index);
                  const auto &x_q    = fe_face_values.quadrature_point(q_index);
                  const auto  comp_i = fe.system_to_component_index(i).first;

                  cell_rhs(i) +=
                    (phi_i[comp_i] * par.neumann_function.value(x_q, comp_i) *
                     fe_face_values.JxW(q_index));
                }
          }


      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}
```

### Running the Simulation

```cpp
int main()
{
  {
    LinearElasticityParameters<2> par;
    LinearElasticity<2>           laplace_problem_2d(par);
    laplace_problem_2d.run();
  }

  return 0;
}
```
