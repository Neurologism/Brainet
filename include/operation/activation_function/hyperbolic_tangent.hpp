#ifndef HYPERBOLICTANGENT_HPP
#define HYPERBOLICTANGENT_HPP

#include "activation_function.hpp"

/**
 * @brief Hyperbolic tangent function class, representing the hyperbolic tangent function f(x) = tanh(x).
*/
class HyperbolicTangent : public ActivationFunction
{   
    /**
     * @brief The hyperbolic tangent function.
     * @param input The input value.
     */
    double activationFunction(double input)override;
    /**
     * @brief The derivative of the hyperbolic tangent function.
     * @param input The input value.
     */
    double activationFunctionDerivative(double input)override;
public:
    HyperbolicTangent() { mName = "HYPERBOLIC_TANGENT"; };
    ~HyperbolicTangent() = default;
};

#endif // HYPERBOLICTANGENT_HPP