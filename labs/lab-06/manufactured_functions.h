#ifndef NMPDE_LAB06_MANUFACTURED_FUNCTIONS_H
#define NMPDE_LAB06_MANUFACTURED_FUNCTIONS_H

#include <deal.II/base/function.h>

#include <cmath>

// Supplied analytical data. Exercise changes belong in poisson-starter.cc.
namespace Lab06
{
  class Affine : public dealii::Function<2>
  {
  public:
    double
    value(const dealii::Point<2> &p, const unsigned int = 0) const override
    {
      return 1. + p[0] + p[1];
    }
    dealii::Tensor<1, 2>
    gradient(const dealii::Point<2> &, const unsigned int = 0) const override
    {
      dealii::Tensor<1, 2> result;
      result[0] = 1.;
      result[1] = 1.;
      return result;
    }
    double
    laplacian(const dealii::Point<2> &, const unsigned int = 0) const override
    {
      return 0.;
    }
  };

  class TwoPeaks : public dealii::Function<2>
  {
  public:
    double
    value(const dealii::Point<2> &p, const unsigned int = 0) const override
    {
      return 1. + .2 * p[0] - .1 * p[1] + peak(p, 0) + peak(p, 1);
    }
    dealii::Tensor<1, 2>
    gradient(const dealii::Point<2> &p, const unsigned int = 0) const override
    {
      dealii::Tensor<1, 2> result;
      result[0] = .2;
      result[1] = -.1;
      for (unsigned int k = 0; k < 2; ++k)
        result += -2. * alpha[k] * peak(p, k) * (p - centers[k]);
      return result;
    }
    double
    laplacian(const dealii::Point<2> &p, const unsigned int = 0) const override
    {
      double result = 0.;
      for (unsigned int k = 0; k < 2; ++k)
        result += (4. * alpha[k] * alpha[k] * p.distance_square(centers[k]) -
                   4. * alpha[k]) *
                  peak(p, k);
      return result;
    }

  private:
    double
    peak(const dealii::Point<2> &p, const unsigned int k) const
    {
      return amplitudes[k] *
             std::exp(-alpha[k] * p.distance_square(centers[k]));
    }
    const dealii::Point<2> centers[2]    = {{-.35, -.2}, {.4, .35}};
    const double           alpha[2]      = {12., 25.};
    const double           amplitudes[2] = {2., 1.2};
  };
} // namespace Lab06
#endif
