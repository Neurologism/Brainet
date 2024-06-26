#ifndef ACTIVATION_FUNCTION_INCLUDE_GUARD
#define ACTIVATION_FUNCTION_INCLUDE_GUARD

#include"..\operation.h"

class ACTIVATION_FUNCTION : public OPERATION
{
public:
    void f(std::vector<VARIABLE *>& inputs) override;
    TENSOR<double> bprop(std::vector<VARIABLE *>& inputs, VARIABLE & focus, TENSOR<double> & gradient) override;

    virtual double activation_function(double x) = 0;
    virtual double activation_function_derivative(double x) = 0;
};

void ACTIVATION_FUNCTION::f(std::vector<VARIABLE *>& inputs)
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("ACTIVATION_FUNCTION::f: Invalid number of input variables.");
    }
    
    TENSOR<double> _data(inputs.front()->get_data()->shape());

    for (double data : inputs.front()->get_data()->data()) // apply activation function to all elements
    {
        _data.data().push_back(activation_function(data));
    }

    *(this->get_variable()->get_data()) = _data;
}

TENSOR<double> ACTIVATION_FUNCTION::bprop(std::vector<VARIABLE *>& inputs, VARIABLE & focus, TENSOR<double> & gradient)
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("ACTIVATION_FUNCTION::bprop: Invalid number of input variables.");
    }

    // load derivative of activation into data 
    TENSOR<double> _data(focus.get_data()->shape());

    for (int i=0; i<gradient.size(); i++) // apply activation function derivative to all elements
    {
        _data.data().push_back(activation_function_derivative(focus.get_data()->data()[i])*gradient.data()[i]);
    }

    return _data;
}

#include "rectified_linear_unit.h"

using ACTIVATION_FUNCTION_VARIANT = std::variant<ReLU>;


#endif // ACTIVATION_FUNCTION_INCLUDE_GUARD