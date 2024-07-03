#ifndef RECTIFIED_LINEAR_UNIT_INCLUDE_GUARD
#define RECTIFIED_LINEAR_UNIT_INCLUDE_GUARD

#include "activation_function.h"

/**
 * @brief Rectified linear unit class, representing the ReLU activation function f(x) = max(x, 0).
*/
class ReLU : public ACTIVATION_FUNCTION
{
protected:
    double __gradient; // gradient of left part of the function
    

    double activation_function(double input)override;
    double activation_function_derivative(double input)override;
public:
    ReLU();
    ~ReLU() = default;
};

/**
 * @brief Constructor for the ReLU class.
*/
ReLU::ReLU()
{
    
    __gradient = 0;
}

double ReLU::activation_function(double input)
{
    return input > 0 ? input : __gradient * input;
}

double ReLU::activation_function_derivative(double input)
{
    return input > 0 ? 1 : __gradient;
}

/**
 * @brief Leaky ReLU class, representing the activation function f(x) = max(x, 0) + left_gradient * min(x, 0).
*/
class LeakyReLU : public ReLU
{
public:
    LeakyReLU(double left_gradient);
};

/**
 * @brief Constructor for the LeakyReLU class.
 * @param gradient The gradient of the function for x < 0.
*/
LeakyReLU::LeakyReLU(double gradient)
{
    __gradient = gradient;
}

/**
 * @brief Absolute ReLU class, representing the activation function f(x) = |x|.
*/
class AbsoluteReLU : public ReLU
{
public:
    AbsoluteReLU();
};

/**
 * @brief Constructor for the AbsoluteReLU class.
*/
AbsoluteReLU::AbsoluteReLU()
{
    __gradient = -1;
}

// add parametric RELU
// add Maxout

#endif // RECTIFIED_LINEAR_UNIT_INCLUDE_GUARD