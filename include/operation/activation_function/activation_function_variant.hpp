//
// Created by servant-of-scietia on 20.09.24.
//

#ifndef ACTIVATION_FUNCTION_VARIANT_HPP
#define ACTIVATION_FUNCTION_VARIANT_HPP

#include "rectified_linear_unit.hpp"
#include "hyperbolic_tangent.hpp"
#include "linear.hpp"
#include "heavyside_step.hpp"
#include "sigmoid.hpp"
#include "softmax.hpp"

using ActivationVariant = std::variant<ReLU, HyperbolicTangent, Sigmoid, Linear, Sigmoid, Softmax>;

#endif //ACTIVATION_FUNCTION_VARIANT_HPP
