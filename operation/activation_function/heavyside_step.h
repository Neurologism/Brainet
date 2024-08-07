#ifndef HEAVYSIDE_STEP_INCLUDE_GUARD
#define HEAVYSIDE_STEP_INCLUDE_GUARD

#include "activation_function.h"

/**
 * @brief Heavyside step function class, representing the Heavyside step function f(x) = 1 if x >= 0, 0 otherwise.
*/
class HeavysideStep : public ACTIVATION_FUNCTION
{   
    std::string __dbg_name = "HEAVYSIDE_STEP";
    double activation_function(double input)override;
    double activation_function_derivative(double input)override;
public:
    HeavysideStep() = default;
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

#endif // HEAVYSIDE_STEP_INCLUDE_GUARD