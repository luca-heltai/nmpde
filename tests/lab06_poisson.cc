#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

#include "../labs/lab-06/manufactured_functions.h"
#include "figure_generation.h"

using namespace dealii;

namespace
{
  struct Errors
  {
    double l2, h1;
  };
  const std::filesystem::path assets = NMPDE_ASSET_DIR;

  // Reference calculation for quantitative checks and figure data.
  Errors
  solve(const Function<2> &exact,
        const unsigned int refinement,
        const std::string &name,
        const unsigned int extra_quadrature = 0)
  {
    Triangulation<2> mesh;
    GridGenerator::hyper_cube(mesh, -1., 1.);
    mesh.refine_global(refinement);
    FE_Q<2>       fe(1);
    DoFHandler<2> dofs(mesh);
    dofs.distribute_dofs(fe);
    DynamicSparsityPattern dsp(dofs.n_dofs());
    DoFTools::make_sparsity_pattern(dofs, dsp);
    SparsityPattern sparsity;
    sparsity.copy_from(dsp);
    SparseMatrix<double> matrix(sparsity);
    Vector<double>       rhs(dofs.n_dofs()), solution(dofs.n_dofs());

    QGauss<2>          quadrature(fe.degree + 3 + extra_quadrature);
    FEValues<2>        values(fe,
                       quadrature,
                       update_values | update_gradients |
                         update_quadrature_points | update_JxW_values);
    FullMatrix<double> cell_matrix(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
    Vector<double>     cell_rhs(fe.n_dofs_per_cell());
    std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
    for (const auto &cell : dofs.active_cell_iterators())
      {
        values.reinit(cell);
        cell_matrix = 0.;
        cell_rhs    = 0.;
        for (const auto q : values.quadrature_point_indices())
          {
            const double f = -exact.laplacian(values.quadrature_point(q));
            for (const auto i : values.dof_indices())
              {
                cell_rhs(i) += f * values.shape_value(i, q) * values.JxW(q);
                for (const auto j : values.dof_indices())
                  cell_matrix(i, j) += values.shape_grad(i, q) *
                                       values.shape_grad(j, q) * values.JxW(q);
              }
          }
        cell->get_dof_indices(indices);
        for (const auto i : values.dof_indices())
          {
            rhs[indices[i]] += cell_rhs(i);
            for (const auto j : values.dof_indices())
              matrix.add(indices[i], indices[j], cell_matrix(i, j));
          }
      }

    std::map<types::global_dof_index, double> boundary;
    VectorTools::interpolate_boundary_values(dofs, 0, exact, boundary);
    // Eliminating columns also updates the RHS and preserves symmetry for CG.
    MatrixTools::apply_boundary_values(boundary, matrix, solution, rhs, true);
    SolverControl            control(4000, 1e-12);
    SolverCG<Vector<double>> solver(control);
    solver.solve(matrix, solution, rhs, PreconditionIdentity());
    Vector<double> residual(solution.size());
    const double   norm           = matrix.residual(residual, solution, rhs);
    double         boundary_error = 0.;
    std::map<types::global_dof_index, double> expected_boundary;
    VectorTools::interpolate_boundary_values(dofs, 0, exact, expected_boundary);
    for (const auto &entry : expected_boundary)
      boundary_error = std::max(boundary_error,
                                std::abs(solution[entry.first] - entry.second));
    std::cout << "refinement=" << refinement << " dofs=" << dofs.n_dofs()
              << " residual=" << norm << " boundary error=" << boundary_error
              << '\n';
    AssertThrow(norm < 1e-10 * std::max(1., rhs.l2_norm()),
                ExcMessage("Large residual"));

    EXPECT_LT(boundary_error, 2e-12);
    EXPECT_LT(norm, 1e-10 * std::max(1., rhs.l2_norm()));
    std::filesystem::create_directories(assets);
    const auto vtu_path =
      assets / (name + "-" + std::to_string(refinement) + ".vtu");
    if (extra_quadrature == 0 && CourseFigures::needs_generation(vtu_path))
      {
        DataOut<2> output;
        output.attach_dof_handler(dofs);
        output.add_data_vector(solution, "solution");
        Vector<double> exact_nodes(solution.size());
        VectorTools::interpolate(dofs, exact, exact_nodes);
        output.add_data_vector(exact_nodes, "exact_at_nodes");
        output.build_patches();
        std::ofstream vtu(vtu_path);
        AssertThrow(vtu, ExcMessage("Cannot open VTU file"));
        output.write_vtu(vtu);
      }

    Vector<double> cell_errors(mesh.n_active_cells());
    QGauss<2>      error_quadrature(fe.degree + 3 + extra_quadrature);
    VectorTools::integrate_difference(dofs,
                                      solution,
                                      exact,
                                      cell_errors,
                                      error_quadrature,
                                      VectorTools::L2_norm);
    const double l2 = VectorTools::compute_global_error(mesh,
                                                        cell_errors,
                                                        VectorTools::L2_norm);
    VectorTools::integrate_difference(dofs,
                                      solution,
                                      exact,
                                      cell_errors,
                                      error_quadrature,
                                      VectorTools::H1_norm);
    const double h1         = VectorTools::compute_global_error(mesh,
                                                        cell_errors,
                                                        VectorTools::H1_norm);
    const auto   nodes_path = assets / (name + "-nodes.csv");
    if (refinement == 6 && extra_quadrature == 0 &&
        CourseFigures::needs_generation(nodes_path))
      {
        const auto points =
          DoFTools::map_dofs_to_support_points(MappingQ1<2>(), dofs);
        std::ofstream nodes(nodes_path);
        AssertThrow(nodes, ExcMessage("Cannot open nodal data file"));
        nodes << std::setprecision(16) << "x,y,solution,exact\n";
        for (unsigned int i = 0; i < points.size(); ++i)
          nodes << points.at(i)[0] << ',' << points.at(i)[1] << ','
                << solution[i] << ',' << exact.value(points.at(i)) << '\n';
      }
    return {l2, h1};
  }
} // namespace

TEST(Lab06Poisson, AffinePatchAndAnalyticalDerivatives)
{
  Lab06::Affine affine;
  const auto    error = solve(affine, 2, "affine");
  EXPECT_LT(error.l2, 1e-11);
  EXPECT_LT(error.h1, 1e-11);
  // Check the supplied derivatives independently by centered differences.
  Lab06::TwoPeaks peaks;
  for (const Point<2> p :
       {Point<2>(-.35, -.2), Point<2>(.13, .27), Point<2>(1., -.4)})
    {
      const double delta     = 1e-4;
      double       laplacian = 0.;
      for (unsigned int d = 0; d < 2; ++d)
        {
          auto plus = p, minus = p;
          plus[d] += delta;
          minus[d] -= delta;
          EXPECT_NEAR((peaks.value(plus) - peaks.value(minus)) / (2. * delta),
                      peaks.gradient(p)[d],
                      2e-6);
          laplacian +=
            (peaks.value(plus) - 2. * peaks.value(p) + peaks.value(minus)) /
            (delta * delta);
        }
      EXPECT_NEAR(laplacian, peaks.laplacian(p), 3e-5);
    }
}

TEST(Lab06Poisson, ConvergenceAndFigureData)
{
  Functions::ExpFunction<2> exponential;
  Lab06::TwoPeaks           peaks;
  for (const auto &name : {std::string("exponential"), std::string("peaks")})
    {
      const Function<2> &exact =
        name == "peaks" ? static_cast<const Function<2> &>(peaks) :
                          static_cast<const Function<2> &>(exponential);
      const auto csv = assets / (name + "-errors.csv");
      std::filesystem::create_directories(assets);
      std::ofstream table;
      if (CourseFigures::needs_generation(csv))
        {
          table.open(csv);
          ASSERT_TRUE(table.good());
        }
      if (table.is_open())
        table << std::setprecision(16) << "degree,h,l2,h1\n";
      Errors previous{0., 0.};
      for (unsigned int r = 2; r <= 6; ++r)
        {
          const auto error = solve(exact, r, name);
          ASSERT_GT(error.l2, 0.);
          ASSERT_GT(error.h1, error.l2);
          if (r > 2)
            {
              EXPECT_LT(error.l2, previous.l2);
              EXPECT_LT(error.h1, previous.h1);
            }
          if (r == 6)
            {
              const double rate_l2 = std::log2(previous.l2 / error.l2);
              const double rate_h1 = std::log2(previous.h1 / error.h1);
              std::cout << name << " rates: L2=" << rate_l2 << " H1=" << rate_h1
                        << '\n';
              EXPECT_NEAR(rate_l2, 2., .12);
              EXPECT_NEAR(rate_h1, 1., .12);
              const auto finer_quadrature = solve(exact, r, name, 1);
              EXPECT_NEAR(error.l2, finer_quadrature.l2, 1e-5 * error.l2);
              EXPECT_NEAR(error.h1, finer_quadrature.h1, 1e-5 * error.h1);
            }
          if (table.is_open())
            table << "1," << 2. / (1u << r) << ',' << error.l2 << ','
                  << error.h1 << '\n';
          previous = error;
        }
    }
}
