#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "activation_function.hpp"

/**
 * @brief Sigmoid function class, representing the sigmoid function f(x) = 1 / (1 + exp(-x)).
*/
class Sigmoid : public ActivationFunction
{   
    double activationFunction(double input)override;
    double activationFunctionDerivative(double input)override;
public:
    Sigmoid() { mName = "SIGMOID"; };
    ~Sigmoid() = default;
};

double Sigmoid::activationFunction(double input)
{
    return 1 / (1 + exp(-input));
}

double Sigmoid::activationFunctionDerivative(double input)
{
    return activationFunction(input) * (1 - activationFunction(input));
}

#endif // SIGMOID_HPP