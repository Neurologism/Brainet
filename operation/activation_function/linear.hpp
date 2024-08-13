#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "activation_function.hpp"

/**
 * @brief Linear function class, representing the linear function f(x) = x.
*/
class Linear : public ActivationFunction
{   
    double activationFunction(double input)override;
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