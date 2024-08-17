#ifndef HEAVYSIDESTEP_HPP
#define HEAVYSIDESTEP_HPP

#include "activation_function.hpp"

/**
 * @brief Heavyside step function class, representing the Heavyside step function f(x) = 1 if x >= 0, 0 otherwise.
*/
class HeavysideStep : public ActivationFunction
{   
    /**
     * @brief The Heavyside step function.
     * @param input The input value.
     */
    double activationFunction(double input)override;
    /**
     * @brief The derivative of the Heavyside step function.
     * @param input The input value.
     */
    double activationFunctionDerivative(double input)override;
public:
    HeavysideStep() { mName = "HEAVYSIDE_STEP"; };
    ~HeavysideStep() = default;
};

double HeavysideStep::activationFunction(double input)
{
    return input >= 0 ? 1 : 0;
}

double HeavysideStep::activationFunctionDerivative(double input)
{
    return 0;
}

#endif // HEAVYSIDESTEP_HPP