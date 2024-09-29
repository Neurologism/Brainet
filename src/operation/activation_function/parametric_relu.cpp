//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/activation_function/parametric_relu.hpp"

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
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(inputs[1]->getData()->shape());

    for(std::uint32_t i = 0; i < inputs[1]->getData()->capacity(); i++)
    {
        double input = inputs[1]->getData()->at(i);
        result->set(i, input >= 0 ? input : slope * input);
    }

    // store the result
    this->getVariable()->getData() = result;
}

std::shared_ptr<Tensor> ParametricReLU::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor> & gradient)
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
        return std::make_shared<Tensor>(Tensor({1},sum));
    }
    if(focus == inputs[1])
    {
        // calculate the gradient of the input
        double slope = inputs[0]->getData()->at(0);
        std::shared_ptr<Tensor> result = std::make_shared<Tensor>(inputs[1]->getData()->shape());

        for(std::uint32_t i = 0; i < gradient->capacity(); i++)
        {
            double input = inputs[1]->getData()->at(i);
            result->set(i, input >= 0 ? gradient->at(i) : slope * gradient->at(i));
        }
        // store the result
        return result;
    }


    throw std::runtime_error("ParametricReLU operation has no focus");

}