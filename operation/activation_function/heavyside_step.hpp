#ifndef HEAVYSIDESTEP_HPP
#define HEAVYSIDESTEP_HPP

#include "activation_function.hpp"

/**
 * @brief Heavyside step function class, representing the Heavyside step function f(x) = 1 if x >= 0, 0 otherwise.
*/
class HeavysideStep : public ActivationFunction
{   
    double activation_function(double input)override;
    double activation_function_derivative(double input)override;
public:
    HeavysideStep() { __dbg_name = "HEAVYSIDE_STEP"; };
    ~HeavysideStep() = default;
};

double HeavysideStep::activation_function(double input)
{
    return input >= 0 ? 1 : 0;
}

double HeavysideStep::activation_function_derivative(double input)
{
    return 0;
}

#endif // HEAVYSIDESTEP_HPP