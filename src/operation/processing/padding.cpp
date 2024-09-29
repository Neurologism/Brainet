//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/processing/padding.hpp"

Padding::Padding(std::uint32_t x_padding, std::uint32_t y_padding, double padding_value)
{
    _x_padding = x_padding;
    _y_padding = y_padding;
    _padding_value = padding_value;
    mName = "PADDING";
}

void Padding::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("Padding::f: Invalid number of input variables.");
    }

    // create a new tensor with the new size and copy the data from the input tensor
    std::shared_ptr<Tensor> _data = std::make_shared<Tensor>(Tensor({inputs.front()->getData()->shape(0) + _x_padding, inputs.front()->getData()->shape(1) + _y_padding}, _padding_value));

    for (std::uint32_t i = 0; i < inputs.front()->getData()->shape(0); i++)
    {
        for (std::uint32_t j = 0; j < inputs.front()->getData()->shape(1); j++)
        {
            _data->set({i, j}, inputs.front()->getData()->at({i, j}));
        }
    }
    this->getVariable()->getData() = _data;
};

std::shared_ptr<Tensor> Padding::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor> & gradient)
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("Padding::bprop: Invalid number of input variables.");
    }

    // create a new tensor with the new size and copy the selected data from the gradient tensor
    std::shared_ptr<Tensor> _data = std::make_shared<Tensor>(inputs.front()->getData()->shape());

    for (std::uint32_t i = 0; i < inputs.front()->getData()->shape(0); i++)
    {
        for (std::uint32_t j = 0; j < inputs.front()->getData()->shape(1); j++)
        {
            _data->set({i, j}, gradient->at({i, j}));
        }
    }

    return _data;
};