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
    std::shared_ptr<Matrix> matrix = std::make_shared<Matrix>(Matrix({inputs.front()->getData()->shape(0) + _x_padding, inputs.front()->getData()->shape(1) + _y_padding}, _padding_value));
    Matrix &matrix_ref = *matrix;
    Matrix &input_ref = *std::static_pointer_cast<Matrix>(inputs.front()->getData());
    for (std::uint32_t i = 0; i < inputs.front()->getData()->shape(0); i++)
    {
        const std::uint32_t shape = inputs.front()->getData()->shape(1);
        for (std::uint32_t j = 0; j < shape; j++)
        {
            matrix_ref.set(i, j, input_ref.at(i, j));
        }
    }
    this->getVariable()->getData() = matrix;
};

std::shared_ptr<Tensor> Padding::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor> & gradient)
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("Padding::bprop: Invalid number of input variables.");
    }

    // create a new tensor with the new size and copy the selected data from the gradient tensor
    std::shared_ptr<Matrix> matrix = std::make_shared<Matrix>(inputs.front()->getData()->shape());
    Matrix &matrix_ref = *matrix;
    Matrix &gradient_ref = *std::static_pointer_cast<Matrix>(gradient);
    for (std::uint32_t i = 0; i < inputs.front()->getData()->shape(0); i++)
    {
        const std::uint32_t shape = inputs.front()->getData()->shape(1);
        for (std::uint32_t j = 0; j < shape; j++)
        {
            matrix_ref.set(i, j, gradient_ref.at(i, j));
        }
    }

    return matrix;
};