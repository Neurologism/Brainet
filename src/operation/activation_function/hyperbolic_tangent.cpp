//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/activation_function/hyperbolic_tangent.hpp"

double HyperbolicTangent::activationFunction(double input)
{
    return tanh(input);
}

double HyperbolicTangent::activationFunctionDerivative(double input)
{
    return 1 - pow(activationFunction(input), 2);
}