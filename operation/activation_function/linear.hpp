#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "activation_function.hpp"

/**
 * @brief Linear function class, representing the linear function f(x) = x.
*/
class Linear : public ActivationFunction
{   
    double activation_function(double input)override;
    double activation_function_derivative(double input)override;
public:
    Linear() { mName = "LINEAR"; };
    ~Linear() = default;
};

double Linear::activation_function(double input)
{
    return input;
}

double Linear::activation_function_derivative(double input)
{
    return 1;
}

#endif // LINEAR_HPP