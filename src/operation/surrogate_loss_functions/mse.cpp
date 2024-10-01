//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/surrogate_loss_functions/mse.hpp"

void MSE::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    // security checks
    if(inputs.size() != 2)
    {
        throw std::runtime_error("MSE operation requires 2 inputs");
    }
    if(inputs[0]->getData()->shape() != inputs[1]->getData()->shape())
    {
        throw std::runtime_error("MSE operation requires inputs to have the same shape");
    }
    if(inputs[0]->getData()->dimensionality() != 2)
    {
        throw std::runtime_error("MSE::f: Other than 2D tensors are not supported");
    }

    // calculate the mean squared error
    double sum = 0;
    for(std::uint32_t i = 0; i < inputs[0]->getData()->capacity(); i++)
    {
        sum += pow(inputs[0]->getData()->at(i) - inputs[1]->getData()->at(i), 2)/2;
    }
    sum /= inputs[0]->getData()->capacity();
    // store the result
    this->getVariable()->getData() = std::make_shared<Tensor>(Tensor({1},sum));
}


std::shared_ptr<Tensor> MSE::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor> & gradient)
{
    // security checks
    if(inputs.size() != 2)
    {
        throw std::runtime_error("MSE operation requires 2 inputs");
    }
    if(inputs[0]->getData()->shape() != inputs[1]->getData()->shape())
    {
        throw std::runtime_error("MSE operation requires inputs to have the same shape");
    }
    if(gradient->shape() != std::vector<size_t>({1}))
    {
        throw std::runtime_error("MSE operation requires gradient to have shape {1}");
    }

    // calculate the gradient of the mean squared error function
    std::shared_ptr<Tensor> _gradient = std::make_shared<Tensor>(Tensor(inputs[0]->getData()->shape()));

    for(std::uint32_t i = 0; i < inputs[0]->getData()->capacity(); i++)
    {
        _gradient->set(i, -(inputs[0]->getData()->at(i) - inputs[1]->getData()->at(i)) / inputs[0]->getData()->shape(1)); // only divide by the size of 1 training example
    }

    return _gradient;
}