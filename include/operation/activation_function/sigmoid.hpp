#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "activation_function.hpp"

/**
 * @brief Sigmoid function class, representing the sigmoid function f(x) = 1 / (1 + exp(-x)).
*/
class Sigmoid : public ActivationFunction
{   
    /**
     * @brief The sigmoid function.
     * @param input The input value.
    */
    double activationFunction(double input)override;
    /**
     * @brief The derivative of the sigmoid function.
     * @param input The input value.
    */
    double activationFunctionDerivative(double input)override;
public:
    Sigmoid() { mName = "SIGMOID"; };
    ~Sigmoid() = default;
};

#endif // SIGMOID_HPP