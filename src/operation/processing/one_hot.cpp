//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/processing/one_hot.hpp"

OneHot::OneHot(std::uint32_t size, double on_value, double off_value)
{
    _size = size;
    _on_value = on_value;
    _off_value = off_value;
    mName = "ONE_HOT";
}

void OneHot::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    // might try assigning input values to indices in the future
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("OneHot::f: Invalid number of input variables.");
    }

    std::shared_ptr<Tensor<double>> _data = std::make_shared<Tensor<double>>(Tensor<double>({inputs.front()->getData()->shape(0), _size}, _off_value)); // create a new tensor to store the result

    for (std::uint32_t i = 0; i < inputs.front()->getData()->shape(0); i++)
    {
        if (inputs.front()->getData()->at({i, 0}) >= _size)
        {
            throw std::invalid_argument("OneHot::f: Input value is larger than the size of the one hot encoding.");
        }
        _data->set({i, static_cast<std::uint32_t>(inputs.front()->getData()->at({i, 0}))}, _on_value);
    }

    this->getVariable()->getData() = _data; // store the result in the variable
}

std::shared_ptr<Tensor<double>> OneHot::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient)
{
    throw std::invalid_argument("OneHot::bprop: Backward pass is not supported for one hot encoding.");
}