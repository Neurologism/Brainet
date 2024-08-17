#ifndef COSTFUNCTION_HPP
#define COSTFUNCTION_HPP

// this might turn into something simliar to the activation function header


#include "mse.hpp" // variant for all cost functions
#include "cross_entropy.hpp"

using CostVariant = std::variant<MSE, CrossEntropy>;

#endif // COST_HPP