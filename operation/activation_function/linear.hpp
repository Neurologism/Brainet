#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "activation_function.hpp"

/**
 * @brief Linear function class, representing the linear function f(x) = x.
*/
class Linear : public ActivationFunction
{   
    /**
     * @brief The linear function.
     * @param input The input value.
     */
    double activationFunction(double input)override;
    /**
     * @brief The derivative of the linear function.
     * @param input The input value.
     */
    double activationFunctionDerivative(double input)override;
public:
    Linear() { mName = "LINEAR"; };
    ~Linear() = default;
};

double Linear::activationFunction(double input)
{
    return input;
}

double Linear::activationFunctionDerivative(double input)
{
    return 1;
}

#endif // LINEAR_HPP