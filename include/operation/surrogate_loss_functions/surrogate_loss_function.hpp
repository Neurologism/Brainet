#ifndef SURROGATE_LOSS_FUNCTION_HPP
#define SURROGATE_LOSS_FUNCTION_HPP

#include "mse.hpp" // variant for all cost functions
#include "mean_absolute_error.hpp"
#include "cross_entropy.hpp"

using SurrogateLossFunctionVariant = std::variant<MSE, CrossEntropy>;

#endif // SURROGATE_LOSS_FUNCTION_HPP