# Final projects

## Seminars

### Seminars based on articles

Your presentation should include all proofs, and a brief introduction for those topics that we did not do in class.

1. [Babuska $\Longleftrightarrow$ Brezzi?](https://www.oden.utexas.edu/media/reports/2006/0608.pdf)

2. [On some techniques for approximating boundary conditions in the finite element method](https://doi.org/10.1016/0377-0427(95)00057-7)

3. [Analysis of the Shifted Boundary Method for the Poisson Problem in General Domains](https://arxiv.org/pdf/2006.00872)

4. [Static Condensation, Hybridization, and HDG Methods](https://epubs.siam.org/doi/epdf/10.1137/070706616)

5. [A unified study of continuous and discontinuous Galerkin methods](https://link.springer.com/article/10.1007/s11425-017-9341-1)

### Seminars based on advanced theory

Your task is to prepare a one-hour lecture on one of the following topics:

1. Drift-diffusion-reaction problems (chap. 8 of {cite:ts}`QuarteroniValli1994`)
    - Well posedness
    - Discretization
    - Inconsistent stabilizations (i.e., artificial diffusion)
    - Consistent stabilizations (i.e., GLS, SUPG)

2. A Posteriori Error Estimates and Adaptive Meshes (chap. 10 of {cite:ts}`ErnGuermond2004`)
    - (A review of) Residual-Based Error Estimates (done in class)
    - Hierarchical Error Estimates
    - Duality-Based Error Estimates

## Coding projects

1. Implement the [shifted boundary method](https://arxiv.org/pdf/2006.00872) (starting from step-60)

2. Implement the [SQH scheme](https://doi.org/10.1080/01630563.2019.1599911) to Solve Nonsmooth PDE Optimal Control Problems

3. Implement a coupled Laplace-Beltrami -- Poisson problem (step-38 + step-6)

4. Add support for Chapel in deal.II* (step-64 con chapel)

5. Implement step-64 using Kokkos.

6. A time dependent problem, using SUNDIALS, and Nitsche boundary conditions.
