//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/activation_function/activation_function.hpp"

void ActivationFunction::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    // always check for the right number of inputs
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("ActivationFunction::f: Invalid number of input variables.");
    }

    std::shared_ptr<Tensor> _data = std::make_shared<Tensor>(inputs.front()->getData()->shape()); // create a new tensor to store the result

    for (std::uint32_t i=0; i < _data->capacity(); i++) // apply activation function to all elements
    {
        _data->set(i, activationFunction(inputs.front()->getData()->at(i))); // apply activation function
    }

    this->getVariable()->getData() = _data; // store the result in the variable
}

std::shared_ptr<Tensor> ActivationFunction::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor> & gradient)
{
    // always check for the right number of inputs
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("ActivationFunction::bprop: Invalid number of input variables.");
    }

    // load derivative of activation into data 
    std::shared_ptr<Tensor> _data = std::make_shared<Tensor>(focus->getData()->shape());

    for (std::uint32_t i=0; i < _data->capacity(); i++) // apply derivative of activation function to all elements
    {
        _data->set(i, activationFunctionDerivative(inputs.front()->getData()->at(i)) * gradient->at(i)); // apply derivative of activation function
    }

    return _data;
}