//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/surrogate_loss_functions/mean_absolute_error.hpp"

void MeanAbsoluteError::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    // security checks
    if(inputs.size() != 2)
    {
        throw std::runtime_error("MeanAbsoluteError operation requires 2 inputs");
    }
    if(inputs[0]->getData()->shape() != inputs[1]->getData()->shape())
    {
        throw std::runtime_error("MeanAbsoluteError operation requires inputs to have the same shape");
    }
    if(inputs[0]->getData()->dimensionality() != 2)
    {
        throw std::runtime_error("MeanAbsoluteError::f: Other than 2D tensors are not supported");
    }

    // calculate the mean absolute error
    double sum = 0;
    for(std::uint32_t i = 0; i < inputs[0]->getData()->capacity(); i++)
    {
        sum += abs(inputs[0]->getData()->at(i) - inputs[1]->getData()->at(i));
    }
    sum /= inputs[0]->getData()->capacity();
    // store the result
    this->getVariable()->getData() = std::make_shared<Tensor>(Tensor({1},sum));
}

std::shared_ptr<Tensor> MeanAbsoluteError::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor> & gradient)
{
    // security checks
    if(inputs.size() != 2)
    {
        throw std::runtime_error("MeanAbsoluteError operation requires 2 inputs");
    }
    if(inputs[0]->getData()->shape() != inputs[1]->getData()->shape())
    {
        throw std::runtime_error("MeanAbsoluteError operation requires inputs to have the same shape");
    }
    if(inputs[0]->getData()->dimensionality() != 2)
    {
        throw std::runtime_error("MeanAbsoluteError::bprop: Other than 2D tensors are not supported");
    }

    // calculate the gradient
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(inputs[0]->getData()->shape());
    for(std::uint32_t i = 0; i < inputs[0]->getData()->capacity(); i++)
    {
        if(inputs[0]->getData()->at(i) > inputs[1]->getData()->at(i))
        {
            result->set(i, -gradient->at(0)/inputs[0]->getData()->shape(1));
        }
        else if(inputs[0]->getData()->at(i) < inputs[1]->getData()->at(i))
        {
            result->set(i, gradient->at(0)/inputs[0]->getData()->shape(1));
        }
        else
        {
            result->set(i, 0);
        }
    }
    return result;
}