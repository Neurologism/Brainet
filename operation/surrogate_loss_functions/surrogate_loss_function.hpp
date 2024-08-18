#ifndef SURROGATE_COST_FUNCTION_HPP
#define SURROGATE_COST_FUNCTION_HPP

// this might turn into something simliar to the activation function header


#include "mse.hpp" // variant for all cost functions
#include "cross_entropy.hpp"

using SurrogateCostVariant = std::variant<MSE, CrossEntropy>;

#endif // SURROGATE_COST_FUNCTION_HPP