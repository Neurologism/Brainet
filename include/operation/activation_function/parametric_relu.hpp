#ifndef PARAMETRIC_RELU_HPP
#define PARAMETRIC_RELU_HPP

#include "operation/operation.hpp"

/**
 * @brief Parametric rectified linear unit class, representing the PReLU activation function f(x) = max(x, 0) + a * min(x, 0).
 * @note The PReLU activation function is a variant of the ReLU activation function that allows the negative slope to be learned.
 */
class ParametricReLU : public Operation
{
public:
    ParametricReLU() { mName = "ParametricReLU"; }
    /**
     * @brief Applies the PReLU activation function to the input tensor.
     * @param inputs The input tensor.
     * @note Assumes that the slope Variable is first in the inputs vector.
     */
    void f(std::vector<std::shared_ptr<Variable>>& inputs) override;
    /**
     * @brief Applies the derivative of the PReLU activation function to the gradient tensor.
     * @param inputs The input tensor.
     * @param focus The focus variable.
     * @param gradient The gradient tensor.
     * @return The gradient tensor.
     */
    std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient) override;
};

#endif // PARAMETRIC_RELU_HPP