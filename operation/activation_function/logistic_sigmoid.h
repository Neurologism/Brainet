#ifndef LOGISTIC_SIGMOID_INCLUDE_GUARD
#define LOGISTIC_SIGMOID_INCLUDE_GUARD

#include "activation_function.h"

/**
 * @brief Logistic sigmoid function class, representing the logistic sigmoid function f(x) = 1 / (1 + exp(-x)).
*/
class LogisticSigmoid : public ACTIVATION_FUNCTION
{   
    double activation_function(double input)override;
    double activation_function_derivative(double input)override;
public:
    LogisticSigmoid() = default;
    ~LogisticSigmoid() = default;
};

double LogisticSigmoid::activation_function(double input)
{
    return 1 / (1 + exp(-input));
}

double LogisticSigmoid::activation_function_derivative(double input)
{
    return activation_function(input) * (1 - activation_function(input));
}

#endif // LOGISTIC_SIGMOID_INCLUDE_GUARD