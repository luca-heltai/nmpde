#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace fs = std::filesystem;

/** A small, dimension-independent point class.
 *
 * The dimension is a compile-time parameter. The Number parameter makes the
 * same class usable with, for example, float or double coordinates.
 */
template <int dim, typename Number = double>
class Point
{
public:
  Number &
  operator[](const std::size_t index)
  {
    return coordinates_.at(index);
  }

  const Number &
  operator[](const std::size_t index) const
  {
    return coordinates_.at(index);
  }

private:
  std::array<Number, dim> coordinates_{};
};


/** Evaluate a simple scalar field on a two-dimensional point. */
template <int dim, typename Number>
Number
scalar_field(const Point<dim, Number> &point)
{
  static_assert(dim == 2, "This first example is two-dimensional.");

  constexpr Number pi = static_cast<Number>(3.14159265358979323846);
  return std::sin(pi * point[0]) * std::sin(pi * point[1]);
}


int
main(int argc, char **argv)
{
  constexpr int         dim             = 2;
  constexpr std::size_t points_per_axis = 21;
  const fs::path output_path = argc > 1 ? fs::path(argv[1]) : "field.csv";

  if (output_path.has_parent_path())
    fs::create_directories(output_path.parent_path());

  std::ofstream output(output_path);
  if (!output)
    {
      std::cerr << "Could not open " << output_path << '\n';
      return 1;
    }

  output << "x,y,u\n";
  output << std::setprecision(16);

  for (std::size_t j = 0; j < points_per_axis; ++j)
    for (std::size_t i = 0; i < points_per_axis; ++i)
      {
        Point<dim> point;
        point[0] = static_cast<double>(i) / (points_per_axis - 1);
        point[1] = static_cast<double>(j) / (points_per_axis - 1);

        output << point[0] << ',' << point[1] << ',' << scalar_field(point)
               << '\n';
      }

  std::cout << "Generated " << points_per_axis * points_per_axis
            << " samples in " << output_path << '\n';
  std::cout << "Point<" << dim << "> uses double coordinates.\n";
}
