// A self-contained reproduction of the harmonic corner experiment in
// FSI-suite, fsi_book/chapter_2.cc, Example2_1_Pacman_0 through _4.
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

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
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "../figure_generation.h"

using namespace dealii;
namespace fs = std::filesystem;

namespace
{
  constexpr unsigned int cycles = 6;

  class CornerSolution : public Function<2>
  {
  public:
    explicit CornerSolution(const unsigned int case_index)
      : omega((2.0 - case_index / 4.0) * numbers::PI)
      , alpha(numbers::PI / omega)
      , theta0(2.0 * numbers::PI - omega)
    {}

    double
    value(const Point<2> &p, const unsigned int = 0) const override
    {
      return std::pow(p.norm(), alpha) * std::sin(alpha * (angle(p) - theta0));
    }

    Tensor<1, 2>
    gradient(const Point<2> &p, const unsigned int = 0) const override
    {
      AssertThrow(p.norm() > 0, ExcMessage("The corner gradient is singular."));
      const double factor = alpha * std::pow(p.norm(), alpha - 1.0);
      const double phase  = (alpha - 1.0) * angle(p) - alpha * theta0;
      Tensor<1, 2> result;
      result[0] = factor * std::sin(phase);
      result[1] = factor * std::cos(phase);
      return result;
    }

    const double omega;
    const double alpha;

  private:
    double
    angle(const Point<2> &p) const
    {
      // Select [theta0, 2*pi], including the lower ray where y can be +0.
      // Using atan2(-y,-x)+pi directly would depend on the sign of zero.
      const double theta = std::atan2(p[1], p[0]);
      return theta < theta0 - 1e-12 ? theta + 2.0 * numbers::PI : theta;
    }
    const double theta0;
  };

  struct Errors
  {
    double        l2 = 0, h1_seminorm = 0;
    Vector<float> cell_h1;
  };

  Errors
  measure_error(const DoFHandler<2>  &dofs,
                const MappingFE<2>   &mapping,
                const Vector<double> &solution,
                const CornerSolution &exact,
                const unsigned int    order)
  {
    // On corner cells, rotate the reference rule so its collapsed edge maps
    // to the singular vertex. beta=1/alpha regularizes the squared gradient.
    const QDuffy regular(order, 1.0);
    const QDuffy singular(order, 1.0 / exact.alpha);
    std::array<std::unique_ptr<FEValues<2>>, 4> values;
    const UpdateFlags flags = update_values | update_gradients |
                              update_quadrature_points | update_JxW_values;
    values[3] =
      std::make_unique<FEValues<2>>(mapping, dofs.get_fe(), regular, flags);
    const std::array<Point<2>, 3> vertices = {
      {Point<2>(0, 0), Point<2>(1, 0), Point<2>(0, 1)}};
    for (unsigned int vertex = 0; vertex < 3; ++vertex)
      {
        std::vector<Point<2>> points;
        for (const auto &p : singular.get_points())
          points.push_back((1 - p[0] - p[1]) * vertices[vertex] +
                           p[0] * vertices[(vertex + 1) % 3] +
                           p[1] * vertices[(vertex + 2) % 3]);
        values[vertex] =
          std::make_unique<FEValues<2>>(mapping,
                                        dofs.get_fe(),
                                        Quadrature<2>(points,
                                                      singular.get_weights()),
                                        flags);
      }

    Errors errors;
    errors.cell_h1.reinit(dofs.get_triangulation().n_active_cells());
    for (const auto &cell : dofs.active_cell_iterators())
      {
        unsigned int vertex = 3;
        for (unsigned int v = 0; v < 3; ++v)
          if (cell->vertex(v).norm() < 1e-14)
            vertex = v;
        auto &fe = *values[vertex];
        fe.reinit(cell);
        std::vector<double>       u(fe.n_quadrature_points);
        std::vector<Tensor<1, 2>> gradient(fe.n_quadrature_points);
        fe.get_function_values(solution, u);
        fe.get_function_gradients(solution, gradient);
        double cell_h1 = 0;
        for (unsigned int q = 0; q < fe.n_quadrature_points; ++q)
          {
            const auto &p = fe.quadrature_point(q);
            errors.l2 +=
              Utilities::fixed_power<2>(u[q] - exact.value(p)) * fe.JxW(q);
            cell_h1 +=
              (gradient[q] - exact.gradient(p)).norm_square() * fe.JxW(q);
          }
        errors.h1_seminorm += cell_h1;
        errors.cell_h1[cell->active_cell_index()] = std::sqrt(cell_h1);
      }
    errors.l2          = std::sqrt(errors.l2);
    errors.h1_seminorm = std::sqrt(errors.h1_seminorm);
    return errors;
  }

  void
  run_case(const unsigned int case_index)
  {
    const CornerSolution exact(case_index);
    const fs::path       output = fs::path(NMPDE_ASSET_DIR) / "lecture-06" /
                            ("pacman-" + std::to_string(case_index));
    std::vector<fs::path> outputs = {output / "convergence.csv",
                                     output / "metadata.txt",
                                     output / "solution.pvd"};
    for (unsigned int cycle = 0; cycle < cycles; ++cycle)
      outputs.push_back(output /
                        ("solution-" + std::to_string(cycle) + ".vtu"));
    if (std::none_of(outputs.begin(),
                     outputs.end(),
                     CourseFigures::needs_generation))
      GTEST_SKIP()
        << "Complete cached pacman outputs; use [regenerate figures] or "
           "NMPDE_REGENERATE_FIGURES=1 to recompute and validate.";
    fs::create_directories(output);
    Triangulation<2> tria;
    GridIn<2>        reader;
    reader.attach_triangulation(tria);
    std::ifstream mesh(fs::path(NMPDE_SOURCE_DIR) / "tests/pacman/meshes" /
                       ("pacman_" + std::to_string(case_index) + ".msh"));
    AssertThrow(mesh, ExcMessage("Cannot read the pacman mesh."));
    reader.read_msh(mesh);
    // Keep the initial polygon fixed: every refinement is affine.
    tria.set_all_manifold_ids(numbers::flat_manifold_id);
    const FE_SimplexP<2> element(1);
    const MappingFE<2>   mapping(element);
    std::ofstream        csv;
    if (CourseFigures::needs_generation(output / "convergence.csv"))
      {
        csv.open(output / "convergence.csv");
        AssertThrow(csv, ExcMessage("Cannot write convergence data."));
        csv
          << "cycle,cells,dofs,h,L2,H1_seminorm,H1,rate_L2,rate_H1,relative_residual,quadrature_L2,quadrature_H1\n";
        csv << std::setprecision(16);
      }
    std::vector<std::pair<double, std::string>> time_series;
    double previous_l2 = 0, previous_h1 = 0, previous_h = 0;
    for (unsigned int cycle = 0; cycle < cycles; ++cycle)
      {
        if (cycle > 0)
          tria.refine_global(1);
        DoFHandler<2> dofs(tria);
        dofs.distribute_dofs(element);
        DynamicSparsityPattern dynamic(dofs.n_dofs());
        DoFTools::make_sparsity_pattern(dofs, dynamic);
        SparsityPattern sparsity;
        sparsity.copy_from(dynamic);
        SparseMatrix<double> matrix(sparsity);
        Vector<double>       solution(dofs.n_dofs()), rhs(dofs.n_dofs());
        FEValues<2>          fe(mapping,
                       element,
                       QGaussSimplex<2>(1),
                       update_gradients | update_JxW_values);
        FullMatrix<double>   local(3, 3);
        std::vector<types::global_dof_index> indices(3);
        for (const auto &cell : dofs.active_cell_iterators())
          {
            local = 0;
            fe.reinit(cell);
            for (unsigned int q = 0; q < fe.n_quadrature_points; ++q)
              for (unsigned int i = 0; i < 3; ++i)
                for (unsigned int j = 0; j < 3; ++j)
                  local(i, j) +=
                    fe.shape_grad(i, q) * fe.shape_grad(j, q) * fe.JxW(q);
            cell->get_dof_indices(indices);
            matrix.add(indices, local);
          }
        std::map<types::global_dof_index, double> boundary;
        for (const auto id : tria.get_boundary_ids())
          VectorTools::interpolate_boundary_values(
            mapping, dofs, id, exact, boundary);
        MatrixTools::apply_boundary_values(boundary, matrix, solution, rhs);
        SolverControl            control(10000, 5e-13 * rhs.l2_norm());
        SolverCG<Vector<double>> solver(control);
        PreconditionSSOR<SparseMatrix<double>> preconditioner;
        preconditioner.initialize(matrix, 1.2);
        solver.solve(matrix, solution, rhs, preconditioner);
        Vector<double> residual(rhs);
        matrix.vmult(residual, solution);
        residual -= rhs;
        const double relative_residual = residual.l2_norm() / rhs.l2_norm();
        EXPECT_LT(relative_residual, 2e-11);

        const Errors errors = measure_error(dofs, mapping, solution, exact, 8);
        // Reintegrate with a higher order only on the finest mesh. The two
        // norms are checked independently of the linear-system residual.
        double quadrature_l2 = 0, quadrature_h1 = 0;
        if (cycle == cycles - 1 && case_index < 4)
          {
            const Errors check =
              measure_error(dofs, mapping, solution, exact, 12);
            quadrature_l2 = std::abs(errors.l2 / check.l2 - 1);
            quadrature_h1 =
              std::abs(errors.h1_seminorm / check.h1_seminorm - 1);
            EXPECT_LT(quadrature_l2, 5e-4);
            EXPECT_LT(quadrature_h1, 5e-4);
          }
        const double h       = GridTools::maximal_cell_diameter(tria);
        const double rate_l2 = cycle == 0 ? 0 :
                                            std::log(previous_l2 / errors.l2) /
                                              std::log(previous_h / h);
        const double rate_h1 = cycle == 0 ?
                                 0 :
                                 std::log(previous_h1 / errors.h1_seminorm) /
                                   std::log(previous_h / h);
        if (case_index == 4)
          {
            // omega=pi: u=-y is affine, hence exactly representable.
            EXPECT_LT(errors.l2, 2e-10);
            EXPECT_LT(errors.h1_seminorm, 2e-9);
          }
        else if (cycle > 0)
          {
            EXPECT_LT(errors.l2, previous_l2);
            EXPECT_LT(errors.h1_seminorm, previous_h1);
            if (cycle == cycles - 1)
              {
                // These broad windows detect broken convergence, without
                // turning asymptotic reference slopes into exact identities.
                EXPECT_NEAR(rate_h1, exact.alpha, 0.12);
                EXPECT_NEAR(rate_l2, 2 * exact.alpha, 0.25);
              }
          }
        if (csv.is_open())
          csv << cycle << ',' << tria.n_active_cells() << ',' << dofs.n_dofs()
              << ',' << h << ',' << errors.l2 << ',' << errors.h1_seminorm
              << ',' << std::hypot(errors.l2, errors.h1_seminorm) << ','
              << rate_l2 << ',' << rate_h1 << ',' << relative_residual << ','
              << quadrature_l2 << ',' << quadrature_h1 << '\n';
        std::cout << "case=" << case_index << " cycle=" << cycle
                  << " dofs=" << dofs.n_dofs() << " L2=" << errors.l2
                  << " H1-semi=" << errors.h1_seminorm << " rates=" << rate_l2
                  << ',' << rate_h1 << std::endl;

        const std::string filename =
          "solution-" + std::to_string(cycle) + ".vtu";
        if (CourseFigures::needs_generation(output / filename))
          {
            Vector<double> exact_nodes(dofs.n_dofs());
            VectorTools::interpolate(mapping, dofs, exact, exact_nodes);
            Vector<double> nodal_error(solution);
            nodal_error -= exact_nodes;
            DataOut<2> data;
            data.attach_dof_handler(dofs);
            data.add_data_vector(solution, "solution");
            data.add_data_vector(exact_nodes, "exact_nodal_interpolant");
            data.add_data_vector(nodal_error, "nodal_error");
            data.add_data_vector(errors.cell_h1,
                                 "cell_H1_error",
                                 DataOut<2>::type_cell_data);
            data.build_patches(mapping);
            std::ofstream vtu(output / filename);
            data.write_vtu(vtu);
          }
        time_series.emplace_back(cycle, filename);
        previous_l2 = errors.l2;
        previous_h1 = errors.h1_seminorm;
        previous_h  = h;
      }
    if (CourseFigures::needs_generation(output / "solution.pvd"))
      {
        std::ofstream pvd(output / "solution.pvd");
        DataOutBase::write_pvd_record(pvd, time_series);
      }
    if (CourseFigures::needs_generation(output / "metadata.txt"))
      {
        std::ofstream metadata(output / "metadata.txt");
        metadata
          << std::setprecision(16) << "deal.II=" << DEAL_II_PACKAGE_VERSION
          << "\ncase=" << case_index
          << "\nomega/pi=" << exact.omega / numbers::PI
          << "\nalpha=" << exact.alpha << "\ncycles=" << cycles
          << "\nelement=P1\nmapping=affine\n"
          << "forcing=0\nDirichlet=nodal interpolation of the exact solution on all boundaries\n"
          << "assembly_quadrature=QGaussSimplex(1)\nerror_quadrature=QDuffy(8), corner beta=1/alpha\n"
          << "quadrature_check=QDuffy(12) on finest mesh\nsolver=CG, SSOR(1.2), relative tolerance 5e-13\n";
      }
  }

  class Pacman : public testing::TestWithParam<unsigned int>
  {};
  TEST_P(Pacman, ConvergenceAndVtu)
  {
    run_case(GetParam());
  }
  INSTANTIATE_TEST_SUITE_P(CornerAngles,
                           Pacman,
                           testing::Values(0, 1, 2, 3, 4));
} // namespace
