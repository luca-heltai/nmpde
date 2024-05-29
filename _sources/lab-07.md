# Lab: Error Estimation using MeshWorker::mesh_loop

## Overview

In this lab, you will extend the work from Lab 5 to implement the theoretical error estimator and use it for adaptive mesh refinement. You will use `MeshWorker::mesh_loop` to efficiently assemble the error estimator and handle face integrals, especially when dealing with hanging nodes. This exercise will help you understand how to integrate face terms and apply adaptive refinement based on error estimates.

## Exercise: Implementing Advanced Error Estimator

The goal of this exercise is to use `MeshWorker::mesh_loop` to assemble the error estimator defined in class:

\[
\eta_T = h_T \| f + \Delta u_h \|_{L^2(T)} + \sum_{F \in \partial T} \frac12 h_F^{1/2} \| [\nabla u_h] \|_{L^2(F)}
\]

### Starting Point

Start from the provided `experiments.cc` file, which demonstrates the use of `MeshWorker::mesh_loop` to compute face integrals. Your task is to integrate this approach into the error estimation for the Poisson problem.

### Key Concepts

1. **MeshWorker::mesh_loop**:
   - This function simplifies the loop over cells and faces, allowing you to define custom cell and face integrals efficiently.

2. **Error Estimator**:
   - Implement an error estimator that combines volume and face terms, especially useful for adaptive refinement.

3. **Handling Hanging Nodes**:
   - Properly integrate face terms in the presence of hanging nodes using `MeshWorker::mesh_loop`.

### Steps

1. **Define the Error Estimator**:
   - Implement the error estimator using `MeshWorker::mesh_loop` based on the definition provided in class.

2. **Integrate Face Terms**:
   - Use `MeshWorker::mesh_loop` to compute face terms, ensuring correct handling of hanging nodes.

3. **Adaptive Refinement**:
   - Use the error estimates to mark and refine the mesh adaptively.

### Explanation: Using `MeshWorker::mesh_loop` for Error Estimation

`MeshWorker::mesh_loop` is used to handle the assembly of both cell and face integrals efficiently, especially when dealing with hanging nodes. Here are the key steps involved:

1. **Define Scratch and Copy Data**:
   - `MeshWorker::ScratchData` is used to store intermediate data during the loop over cells and faces.
   - `MeshWorker::CopyData` is used to store the local contributions that will be copied to the global system.

2. **Cell Worker Function**:
   - This function assembles the cell integrals by looping over the quadrature points and shape functions.

3. **Face Worker Function**:
   - This function assembles the face integrals, which are important for computing the jump terms in the error estimator.

4. **Copier Function**:
   - This function copies the local contributions to the global system, applying constraints as needed.

5. **Run the Mesh Loop**:
   - `MeshWorker::mesh_loop` is called with the defined worker and copier functions to perform the assembly.

By following these steps and using `MeshWorker::mesh_loop`, you can efficiently assemble the error estimator and handle complex mesh configurations with hanging nodes. This approach simplifies the implementation and ensures accurate computation of both cell and face terms in the error estimator.

### Additional Exercise: Advanced Error Estimation with `MeshWorker::mesh_loop`

1. **Implement the Error Estimator**:
   - Extend the `estimate` method to fully implement the advanced error estimator using `MeshWorker::mesh_loop`.

2. **Integrate with Adaptive Refinement**:
   - Use the computed error estimates to mark cells for refinement and perform adaptive mesh refinement.

3. **Analyze Results**:
   - Compare the results of adaptive refinement with uniform refinement, and analyze the efficiency and accuracy of the solution.
   - Compare your error estimator with the Kelly error estimator. Do you see any noticeable differences?
