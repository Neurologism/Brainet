//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/activation_function/rectified_linear_unit.hpp"

/**
 * @brief Constructor for the ReLU class.
*/
ReLU::ReLU(double gradient)
{
    __gradient = gradient; // gradient of (-inf, 0)
    mName = "RELU";
}

double ReLU::activationFunction(double input)
{
    return input >= 0 ? input : __gradient * input;
}

double ReLU::activationFunctionDerivative(double input)
{
    return input >= 0 ? 1 : __gradient;
}