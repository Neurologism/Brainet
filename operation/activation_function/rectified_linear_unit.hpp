#ifndef RECTIFIEDLINEARUNIT_HPP
#define RECTIFIEDLINEARUNIT_HPP

#include "activation_function.hpp"

/**
 * @brief Rectified linear unit class, representing the ReLU activation function f(x) = max(x, 0).
*/
class ReLU : public ActivationFunction
{
    double __gradient; // gradient of left part of the function

    double activationFunction(double input)override;
    double activationFunctionDerivative(double input)override;
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


// add parametric RELU
// add Maxout

#endif // RECTIFIEDLINEARUNIT_HPP