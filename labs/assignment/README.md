# Assignement 01 - Error Computation

## Numerical Methods for the Solution of PDEs

**Luca Heltai** <luca.heltai@unipi.it>

* * * * *

1. Start the assignment from lab-04 (a modified version of step-4)
    <https://www.dealii.org/current/doxygen/deal.II/step_4.html>

2. Add parameters to the program to compute $L^2$ and $H^1$ errors w.r.t. to a
   "manufactured solution", i.e., a solution where the right hand side is
   constructed artificially to guarantee that the exact solution is a given
   function, i.e., if $u=u_g$, the right hand side should be $-\Delta u_g$
   (where $g$ stands for *given*). This is what was asked during the laboratory
   in class. In particular, the parameter file should contain a variable `number
   of cycles`, and the `run` method should have a `for` loop to repeat the
   computation after refining the grid.
  
   For information on how to output the errors in a more readable way, see
   `step-7` (it is a bit more complicated though)
   <https://www.dealii.org/current/doxygen/deal.II/step_7.html>

3. Fix the right-hand side and boundary conditions to get the manufactured
    solution $$u(x) = \sin(\pi x )\cdot\cos(\pi y)$$ and make sure the $L^2$ and
    the $H^1$ errors are converging for degree 1,2, and 3.

4. Add a parameter to give the possibility to change domain type to L-shaped
   domain, and make sure you can run the code with both `hyper_cube` and
   `hyper_L` domain types (using the functions `GridGenerator::hyper_cube` and
   `GridGenerator::hyper_L`). Make sure the parameter file can read a parameter
   `domain type`, corresponding to the variable `par.domain_type` of type
   `std::string`.

   Find out how to make sure that the parameter can only be chosen between
   `hyper_cube` and `hyper_L` (see the documentation of
   `ParameterHandler::add_parameter()` and of `Patterns::PatternBase`, and pass
   the right `Patterns::*` type. See `step-70` for examples on advanced use of
   the `parameter_handler` class
   <https://www.dealii.org/current/doxygen/deal.II/step_70.html>, for an example
   on how to build a selection (look for the way `refinement_strategy` is
   defined, and use the same for the domain).

5. Consider the case of the `hyper_L`, in dimension 2, with the re-entrant
   corner centered on the origin. Construct a manufactured solution that would
   read in radial coordinates as

   $$ u(r,\theta) = r^\frac{\pi}{\omega}\sin(\frac{\pi}{\omega}(\theta - (2\pi-\omega)) $$

   and choose $\omega$ so that the solution is zero on both sides adjacent to the  re-entrant corner.

   Notice that the solution above is such that the right-hand-side is zero (i.e., it  is harmonic). You will only need to apply boundary conditions.

   Construct the convergence table for the case above for degree 1 and 2. What  convergence rates do you observe? How do you explain this?
