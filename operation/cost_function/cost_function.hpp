#ifndef COST_HPP
#define COST_HPP

// this might turn into something simliar to the activation function header


#include "mse.hpp" // variant for all cost functions


using CostVariant = std::variant<MSE>;

#endif // COST_HPP