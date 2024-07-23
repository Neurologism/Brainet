#ifndef PADDING_INCLUDE_GUARD
#define PADDING_INCLUDE_GUARD

#include "../operation.h"

/**
 * @brief Padding class, used to add padding in positive x and y direction to the input tensor in shape (x, y). Should be extended to work for any amount of dimensions and in both directions.
*/
class Padding : public OPERATION
{
    // store data
    int _x_padding;
    int _y_padding;
    double _padding_value;

public:
    /**
     * @brief Add a padding unit to the graph.
     * @param x_padding The amount of padding in the x direction.
     * @param y_padding The amount of padding in the y direction.
     * @param padding_value The value of the padding. 
     */
    Padding(int x_padding, int y_padding, double padding_value);
    ~Padding() = default;

    /**
     * @brief Add padding to the input tensor.
     */
    virtual void f(std::vector<std::shared_ptr<VARIABLE>>& inputs)override;
    /**
     * @brief Remove padding from the gradient tensor.
     */
    virtual std::shared_ptr<TENSOR<double>> bprop(std::vector<std::shared_ptr<VARIABLE>>& inputs, std::shared_ptr<VARIABLE> & focus, std::shared_ptr<TENSOR<double>> & gradient)override;
};

Padding::Padding(int x_padding, int y_padding, double padding_value)
{
    _x_padding = x_padding;
    _y_padding = y_padding;
    _padding_value = padding_value;
}

void Padding::f(std::vector<std::shared_ptr<VARIABLE>>& inputs)
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("Padding::f: Invalid number of input variables.");
    }

    // create a new tensor with the new size and copy the data from the input tensor
    std::shared_ptr<TENSOR<double>> _data = std::make_shared<TENSOR<double>>(TENSOR<double>({inputs.front()->get_data()->shape(0) + _x_padding, inputs.front()->get_data()->shape(1) + _y_padding}, _padding_value));

    for (int i = 0; i < inputs.front()->get_data()->shape(0); i++)
    {
        for (int j = 0; j < inputs.front()->get_data()->shape(1); j++)
        {
            _data->set({i, j}, inputs.front()->get_data()->at({i, j}));
        }
    }
    this->get_variable()->get_data() = _data;
};

std::shared_ptr<TENSOR<double>> Padding::bprop(std::vector<std::shared_ptr<VARIABLE>>& inputs, std::shared_ptr<VARIABLE> & focus, std::shared_ptr<TENSOR<double>> & gradient)
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("Padding::bprop: Invalid number of input variables.");
    }

    // create a new tensor with the new size and copy the selected data from the gradient tensor
    std::shared_ptr<TENSOR<double>> _data = std::make_shared<TENSOR<double>>(inputs.front()->get_data()->shape());

    for (int i = 0; i < inputs.front()->get_data()->shape(0); i++)
    {
        for (int j = 0; j < inputs.front()->get_data()->shape(1); j++)
        {
            _data->set({i, j}, gradient->at({i, j}));
        }
    }

    return _data;
};

#endif // PADDING_INCLUDE_GUARD