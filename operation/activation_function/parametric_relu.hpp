#ifndef PARAMETRIC_RELU_HPP
#define PARAMETRIC_RELU_HPP

#include "../operation.hpp"

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

void ParametricReLU::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    // security checks
    if(inputs.size() != 2)
    {
        throw std::runtime_error("ParametricReLU operation requires 2 inputs");
    }
    if(inputs[0]->getData()->shape() != std::vector<size_t>({1}))
    {
        throw std::runtime_error("ParametricReLU operation requires the slope to be a scalar");
    }

    // calculate the PReLU activation function
    double slope = inputs[0]->getData()->at(0);
    std::shared_ptr<Tensor<double>> result = std::make_shared<Tensor<double>>(inputs[1]->getData()->shape());

    for(std::uint32_t i = 0; i < inputs[1]->getData()->capacity(); i++)
    {
        double input = inputs[1]->getData()->at(i);
        result->set(i, input >= 0 ? input : slope * input);
    }
    
    // store the result
    this->getVariable()->getData() = result;
}

std::shared_ptr<Tensor<double>> ParametricReLU::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient)
{
    // security checks
    if(inputs.size() != 2)
    {
        throw std::runtime_error("ParametricReLU operation requires 2 inputs");
    }
    if(inputs[0]->getData()->shape() != std::vector<size_t>({1}))
    {
        throw std::runtime_error("ParametricReLU operation requires the slope to be a scalar");
    }

    if(focus == inputs[0])
    {
        // calculate the gradient of the slope
        double sum = 0;
        for(std::uint32_t i = 0; i < gradient->capacity(); i++)
        {
            double input = inputs[1]->getData()->at(i);
            sum += input < 0 ? input * gradient->at(i) : 0;
        }
        // store the result
        this->getVariable()->getData() = std::make_shared<Tensor<double>>(Tensor<double>({1},sum));
    }
    else if(focus == inputs[1])
    {
        // calculate the gradient of the input
        double slope = inputs[0]->getData()->at(0);
        std::shared_ptr<Tensor<double>> result = std::make_shared<Tensor<double>>(inputs[1]->getData()->shape());

        for(std::uint32_t i = 0; i < gradient->capacity(); i++)
        {
            double input = inputs[1]->getData()->at(i);
            result->set(i, input >= 0 ? gradient->at(i) : slope * gradient->at(i));
        }
        // store the result
        this->getVariable()->getData() = result;
    }
    else
    {
        throw std::runtime_error("ParametricReLU operation has no focus");
    }
}

#endif // PARAMETRIC_RELU_HPP