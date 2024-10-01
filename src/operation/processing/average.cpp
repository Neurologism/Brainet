//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/processing/average.hpp"

void Average::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    if (inputs.size() < 2)
    {
        throw std::runtime_error("Average: number of inputs is less than 2");
    }

    std::uint32_t size = inputs[0]->getData()->capacity();
    for (std::uint32_t i = 1; i < inputs.size(); i++)
    {
        if (inputs[i]->getData()->capacity() != size)
        {
            throw std::runtime_error("Average: the size of all inputs must be the same");
        }
    }

    auto result = std::make_shared<Tensor>(Tensor(inputs[0]->getData()->shape()));

    for (std::uint32_t i = 0; i < size; i++)
    {
        double sum = 0;
        for (std::uint32_t j = 0; j < inputs.size(); j++)
        {
            sum += inputs[j]->getData()->at({i});
        }
        result->set({i}, sum / inputs.size());
    }

    this->getVariable()->getData() = result;
}

std::shared_ptr<Tensor> Average::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor> & gradient)
{
    throw std::runtime_error("Average: backward pass is currently not supported");
}