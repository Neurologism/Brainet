#ifndef HYPERBOLICTANGENT_HPP
#define HYPERBOLICTANGENT_HPP

#include "activation_function.hpp"

/**
 * @brief Hyperbolic tangent function class, representing the hyperbolic tangent function f(x) = tanh(x).
*/
class HyperbolicTangent : public ActivationFunction
{   
    double activationFunction(double input)override;
    double activationFunctionDerivative(double input)override;
public:
    HyperbolicTangent() { mName = "HYPERBOLIC_TANGENT"; };
    ~HyperbolicTangent() = default;
};

double HyperbolicTangent::activationFunction(double input)
{
    return tanh(input);
}

double HyperbolicTangent::activationFunctionDerivative(double input)
{
    return 1 - pow(activationFunction(input), 2);
}

#endif // HYPERBOLICTANGENT_HPP