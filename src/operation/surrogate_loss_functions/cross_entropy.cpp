//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/surrogate_loss_functions/cross_entropy.hpp"

void CrossEntropy::f(std::vector<std::shared_ptr<Variable>> &inputs)
{
    if (inputs.size() != 2)
    {
        throw std::runtime_error("CrossEntropy: number of inputs is not 2");
    }

    if (inputs[1]->getData()->shape(1) != 1)
    {
        throw std::runtime_error("CrossEntropy: the target tensor must be 1D");
    }

    if (inputs[0]->getData()->shape(0) != inputs[1]->getData()->shape(0))
    {
        throw std::runtime_error("CrossEntropy: the size of the prediction and target tensor must be the same");
    }

    double error = 0;
    for (std::uint32_t i = 0; i < inputs[0]->getData()->shape(0); i++)
    {
        if (mUseWithLog)
        {
            error -= log(inputs[0]->getData()->at({i, static_cast<std::uint32_t>(inputs[1]->getData()->at({i}))}));
        }
        else
        {
            error -= inputs[0]->getData()->at({i, static_cast<std::uint32_t>(inputs[1]->getData()->at({i}))});
        }
    }

    this->getVariable()->getData() = std::make_shared<Tensor>(Tensor({1}, error / static_cast<double>(inputs[0]->getData()->shape(0))));
}


std::shared_ptr<Tensor> CrossEntropy::bprop(std::vector<std::shared_ptr<Variable>> &inputs, std::shared_ptr<Variable> &focus, std::shared_ptr<Tensor> &gradient)
{
    if (inputs.size() != 2)
    {
        throw std::runtime_error("CrossEntropy: number of inputs is not 2");
    }

    if (inputs[1]->getData()->shape(1) != 1)
    {
        throw std::runtime_error("CrossEntropy: the target tensor must be 1D");
    }

    if (gradient->shape() != std::vector<size_t>({1}))
    {
        throw std::runtime_error("CrossEntropy: the gradient tensor must have shape {1}");
    }

    auto _gradient = std::make_shared<Tensor>(Tensor(inputs[0]->getData()->shape()));

    for (std::uint32_t i = 0; i < inputs[0]->getData()->shape(0); i++)
    {
        for (std::uint32_t j = 0; j < inputs[0]->getData()->shape(1); j++)
        {
            if (mUseWithLog)
            {
                _gradient->set({i, j}, -1 / inputs[0]->getData()->at({i, j}) * (j == static_cast<std::uint32_t>(inputs[1]->getData()->at({i}))) * gradient->at({0})); // the gradient of log(x) is -1/x
            }
            else
            {
                _gradient->set({i, j}, -1*(j == static_cast<std::uint32_t>(inputs[1]->getData()->at({i}))) * gradient->at({0}));
            }
        }
    }

    return _gradient;
}

void CrossEntropy::useWithExp()
{
    mUseWithLog = false;
}