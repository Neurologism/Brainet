//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/activation_function/softmax.hpp"

void Softmax::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    // always check for right number of inputs
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("Softmax::f: Invalid number of input variables.");
    }

    std::shared_ptr<Tensor> _data = std::make_shared<Tensor>(inputs.front()->getData()->shape()); // create a new tensor to store the result

    for (std::uint32_t i = 0; i < inputs.front()->getData()->shape()[0]; i++)
    {

        double _max = inputs.front()->getData()->at({i, 0}); // normalize the input to avoid overflow / underflow
        for (std::uint32_t j = 0; j < inputs.front()->getData()->shape()[1]; j++)
        {
            if (inputs.front()->getData()->at({i, j}) > _max)
            {
                _max = inputs.front()->getData()->at({i, j});
            }
        }


        double _sum = 0;
        for (std::uint32_t j = 0; j < inputs.front()->getData()->shape()[1]; j++)
        {
            _sum += std::exp(inputs.front()->getData()->at({i, j}) - _max);
        }

        if (mUseWithExp)
        {
            for (std::uint32_t j = 0; j < inputs.front()->getData()->shape()[1]; j++)
            {
                _data->set({i, j}, std::exp(inputs.front()->getData()->at({i, j}) - _max) / _sum);
            }
        }
        else
        {
            for (std::uint32_t j = 0; j < inputs.front()->getData()->shape()[1]; j++)
            {
                _data->set({i, j}, inputs[0]->getData()->at({i, j}) - _max - std::log(_sum));
            }
        }
    }
    this->getVariable()->getData() = _data; // store the result in the variable
}

std::shared_ptr<Tensor> Softmax::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor> & gradient)
{
    // always check for right number of inputs
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("Softmax::bprop: Invalid number of input variables.");
    }
    if (inputs.front() != focus)
    {
        throw std::invalid_argument("Softmax::bprop: The focus variable is not the input variable.");
    }

    std::shared_ptr<Tensor> data = this->getVariable()->getData(); // get the data of the variable
    std::shared_ptr<Tensor> grad = std::make_shared<Tensor>(inputs.front()->getData()->shape()); // create a new tensor to store the gradient

    for (std::uint32_t i = 0; i < inputs.front()->getData()->shape()[0]; i++)
    {
        if (mUseWithExp)
        {
            double _sum = 0;
            for (std::uint32_t j = 0; j < inputs.front()->getData()->shape()[1]; j++) // precalculate the sum of the gradient
            {
                _sum += data->at({i, j}) * gradient->at({i, j});
            }

            for (std::uint32_t j = 0; j < inputs.front()->getData()->shape()[1]; j++)
            {
                grad->set({i, j}, data->at({i,j}) * ( gradient->at({i,j}) - _sum ));
            }
        }
        else
        {
            for (std::uint32_t j = 0; j < inputs.front()->getData()->shape()[1]; j++) // simplified version of the gradient
            {
                grad->set({i, j}, std::exp(data->at({i, j})) + gradient->at({i, j}));
            }
        }
    }

    return grad; // return the gradient
}

void Softmax::useWithLog()
{
    mUseWithExp = false;
}