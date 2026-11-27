# Lab: From CG Poisson to DG Poisson (SIPG)

## Overview

In this lab, you will use `MeshWorker::mesh_loop` to efficiently assemble the system matrix and right-hand side, handle boundary contributions using the Nitsche method, and implement face contributions to assemble the Symmetric Interior Penalty Galerkin (SIPG) method. This exercise will help you understand how to integrate cell and face terms and apply adaptive refinement based on error estimates.

## Exercises

### Exercise 1: Replacing the Manually Written Assemble Loop with MeshWorker::mesh_loop

#### Objective

Replace the manually written assemble loop with `MeshWorker::mesh_loop` to handle the assembly process more efficiently.

#### Steps

1. **Setup Scratch and Copy Data:**
   - Use `MeshWorker::ScratchData` and `MeshWorker::CopyData` structures to store intermediate data during the loop over cells and faces.

2. **Define Cell Worker Function:**
   - Create a cell worker function that assembles the local system matrix and right-hand side for CG methods.

3. **Define Copier Function:**
   - Create a copier function to transfer local contributions to the global system.

4. **Run Mesh Loop:**
   - Replace the manual assembly loop with `MeshWorker::mesh_loop` using the defined worker and copier functions.

### Exercise 2: Add Boundary Contributions to assemble boundary terms Using Nitsche Method

#### Objective

Incorporate boundary contributions using the Nitsche method for weakly imposing Dirichlet boundary conditions.

#### Steps

1. **Setup Face Scratch Data:**
   - Initialize `MeshWorker::ScratchData` to include face integration rules and update flags for boundary terms.

2. **Define Boundary Face Worker Function:**
   - Create a boundary face worker function that assembles the boundary contributions using the Nitsche method. Remember, for CG with imposition of boundary conditions with constraints, the weak form looks like

   $$
   (\nabla u, \nabla v) = (f,v) \qquad \forall v \in V
   $$

   while for CG with Nitsche boundary conditions, the weak form looks like:

   $$
   (\nabla u, \nabla v) - <n\cdot \nabla u, v>  - <u, n\cdot \nabla v> +\frac{\gamma}{h}<u,v> = (f,v)
   - <g, n\cdot \nabla v> + \frac{\gamma}{h}<g,v>
   \qquad \forall v \in V
   $$
   where we indicate with $(\cdot, \cdot)$ the $L^2$ scalar product in $\Omega$ and with $<\cdot, \cdot>$ the $L^2$ scalar product on the Dirichlet part of the boundary $\partial \Omega_D$.

3. **Run Mesh Loop:**
   - Extend `MeshWorker::mesh_loop` to include the boundary face worker function for boundary term assembly.

### Exercise 3: Add Face Contributions to the Loop for SIPG Method

#### Objective

Extend the assembly process to include face contributions for the Symmetric Interior Penalty Galerkin (SIPG) method (to read more, take a look at step-12 of the deal.II library).

Remember: SIPG is an extension of the Nitsche method implemented above with the addition of the following face terms:

$$
- <\{\!\{\nabla u\}\!\}, [\![v]\!]>_{\mathcal E^0} - <[\![u]\!], \{\!\{\nabla v\}\!\}>_{\mathcal E^0} +\frac{\gamma}{h}<[\![u]\!],[\![v]\!]>_{\mathcal E^0}
$$
where we indicate with $\mathcal E^0$ the set of the interior faces, and with $<\cdot, \cdot>$ the $L^2$ scalar product on the co-dimension one faces.

#### Steps

1. **Setup Face Scratch Data:**
   - Extend `MeshWorker::ScratchData` to include face integration rules and update flags for face terms.

2. **Define Interior Face Worker Function:**
   - Create an interior face worker function that assembles the SIPG contributions for the interior faces.

3. **Run Mesh Loop:**
   - Extend `MeshWorker::mesh_loop` to include the interior face worker function for SIPG term assembly.
