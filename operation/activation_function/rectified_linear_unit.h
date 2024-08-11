#ifndef RECTIFIED_LINEAR_UNIT_INCLUDE_GUARD
#define RECTIFIED_LINEAR_UNIT_INCLUDE_GUARD

#include "activation_function.h"

/**
 * @brief Rectified linear unit class, representing the ReLU activation function f(x) = max(x, 0).
*/
class ReLU : public ActivationFunction
{
    double __gradient; // gradient of left part of the function

    double activation_function(double input)override;
    double activation_function_derivative(double input)override;
public:
    ReLU(double gradient = 0);
    ~ReLU() = default;
};

/**
 * @brief Constructor for the ReLU class.
*/
ReLU::ReLU(double gradient)
{
    __gradient = gradient; // gradient of (-inf, 0)
    __dbg_name = "RELU";
}

double ReLU::activation_function(double input)
{
    return input >= 0 ? input : __gradient * input;
}

double ReLU::activation_function_derivative(double input)
{
    return input >= 0 ? 1 : __gradient;
}


// add parametric RELU
// add Maxout

#endif // RECTIFIED_LINEAR_UNIT_INCLUDE_GUARD