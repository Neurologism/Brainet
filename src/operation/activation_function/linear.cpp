//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/activation_function/linear.hpp"

double Linear::activationFunction(double input)
{
    return input;
}

double Linear::activationFunctionDerivative(double input)
{
    return 1;
}