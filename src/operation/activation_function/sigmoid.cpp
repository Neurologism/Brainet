//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/activation_function/sigmoid.hpp"

double Sigmoid::activationFunction(double input)
{
    return 1 / (1 + exp(-input));
}

double Sigmoid::activationFunctionDerivative(double input)
{
    return activationFunction(input) * (1 - activationFunction(input));
}