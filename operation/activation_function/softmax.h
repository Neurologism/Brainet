#ifndef SOFTMAX_INCLUDE_GUARD
#define SOFTMAX_INCLUDE_GUARD

#include "activation_function.h"

/**
 * @brief Softmax function class, representing the softmax function f(x) = exp(x) / sum(exp(x)).
*/
class Softmax : public ACTIVATION_FUNCTION
{   
    double activation_function(double input)override;
    double activation_function_derivative(double input)override;
public:
    Softmax() = default;
    ~Softmax() = default;
};

double Softmax::activation_function(double input)
{
    return exp(input);
}

double Softmax::activation_function_derivative(double input)
{
    return activation_function(input) * (1 - activation_function(input));
}

#endif // SOFTMAX_INCLUDE_GUARD