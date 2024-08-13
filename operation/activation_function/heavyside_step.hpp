#ifndef HEAVYSIDESTEP_HPP
#define HEAVYSIDESTEP_HPP

#include "activation_function.hpp"

/**
 * @brief Heavyside step function class, representing the Heavyside step function f(x) = 1 if x >= 0, 0 otherwise.
*/
class HeavysideStep : public ActivationFunction
{   
    double activationFunction(double input)override;
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