//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/activation_function/heavyside_step.hpp"

double HeavysideStep::activationFunction(double input)
{
    return input >= 0 ? 1 : 0;
}

double HeavysideStep::activationFunctionDerivative(double input)
{
    return 0;
}