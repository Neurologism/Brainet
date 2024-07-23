#ifndef COST_FUNCTION_INCLUDE_GUARD
#define COST_FUNCTION_INCLUDE_GUARD

// this might turn into something simliar to the activation function header


#include "mse.h" // variant for all cost functions


using COST_FUNCTION_VARIANT = std::variant<MSE>;

#endif // COST_FUNCTION_INCLUDE_GUARD