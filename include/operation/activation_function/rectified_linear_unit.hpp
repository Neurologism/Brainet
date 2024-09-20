#ifndef RECTIFIEDLINEARUNIT_HPP
#define RECTIFIEDLINEARUNIT_HPP

#include "activation_function.hpp"

/**
 * @brief Rectified linear unit class, representing the ReLU activation function f(x) = max(x, 0).
*/
class ReLU : public ActivationFunction
{
    double __gradient; // gradient of left part of the function

protected:
    /**
     * @brief The ReLU function.
     * @param input The input value.
    */
    double activationFunction(double input)override;
    /**
     * @brief The derivative of the ReLU function.
     * @param input The input value.
    */
    double activationFunctionDerivative(double input)override;
public:
    ReLU(double gradient = 0);
    ~ReLU() = default;
};

// add Maxout

#endif // RECTIFIEDLINEARUNIT_HPP