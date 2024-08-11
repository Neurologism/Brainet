#ifndef HYPERBOLIC_TANGENT_INCLUDE_GUARD
#define HYPERBOLIC_TANGENT_INCLUDE_GUARD

#include "activation_function.h"

/**
 * @brief Hyperbolic tangent function class, representing the hyperbolic tangent function f(x) = tanh(x).
*/
class HyperbolicTangent : public ActivationFunction
{   
    double activation_function(double input)override;
    double activation_function_derivative(double input)override;
public:
    HyperbolicTangent() { __dbg_name = "HYPERBOLIC_TANGENT"; };
    ~HyperbolicTangent() = default;
};

double HyperbolicTangent::activation_function(double input)
{
    return tanh(input);
}

double HyperbolicTangent::activation_function_derivative(double input)
{
    return 1 - pow(activation_function(input), 2);
}

#endif // HYPERBOLIC_TANGENT_INCLUDE_GUARD