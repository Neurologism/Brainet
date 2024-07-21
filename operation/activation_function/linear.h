#ifndef LINEAR_INCLUDE_GUARD
#define LINEAR_INCLUDE_GUARD

#include "activation_function.h"

/**
 * @brief Linear function class, representing the linear function f(x) = x.
*/
class Linear : public ACTIVATION_FUNCTION
{   
    double activation_function(double input)override;
    double activation_function_derivative(double input)override;
public:
    Linear() = default;
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

#endif // LINEAR_INCLUDE_GUARD