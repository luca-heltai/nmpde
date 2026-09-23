# Lab 5d: Nitsche boundary conditions

This lab implements the Nitsche boundary conditions for a PDE problem. The
Nitsche method is a technique used in finite element methods to impose boundary
conditions weakly. It is particularly useful for problems where the boundary
conditions are not easily enforced in a strong sense, such as in the case of
non-homogeneous Dirichlet or Neumann conditions.

Weak form of the problem with Nitsche boundary conditions:

$$
\begin{split}
& \int_{\Omega} C \nabla u \cdot \nabla v \, dx \\
 & - \int_{\Gamma_D} n\cdot C \nabla u v \, ds - \int_{\Gamma_D} n\cdot C\nabla v (u-g) \, ds \\
 & + \gamma \int_{\Gamma_D} (u - g) v \, ds = \int_{\Omega} f v \, dx \qquad \forall v \in V_h
\end{split}
$$

elasticity:

$$
\begin{split}
&-div(\sigma(u)) = f \text{ in } \Omega, \\
& \sigma(u) = C \nabla u = \mu\epsilon(u) +\lambda I div(u) \\
& \epsilon(u) = \frac{1}{2}(\nabla u + (\nabla u)^T) \\
\end{split}
$$
On the boundary
$$
\begin{split}
&\int_\Gamma n\cdot( C\nabla u )v \, ds := \\
& \int_\Gamma \mu n\cdot \epsilon(u) v  + \lambda div(u) n \cdot v
\end{split}
$$
