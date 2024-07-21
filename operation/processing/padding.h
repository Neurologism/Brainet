#ifndef PADDING_INCLUDE_GUARD
#define PADDING_INCLUDE_GUARD

#include "../operation.h"

/**
 * @brief Padding operation class, representing the padding operation.
*/
class Padding : public OPERATION
{
    int _x_padding;
    int _y_padding;
    double _padding_value;

public:
    Padding(int x_padding, int y_padding, double padding_value);
    ~Padding() = default;

    virtual void f(std::vector<std::shared_ptr<VARIABLE>>& inputs)override;
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