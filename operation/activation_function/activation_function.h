#ifndef ACTIVATION_FUNCTION_INCLUDE_GUARD
#define ACTIVATION_FUNCTION_INCLUDE_GUARD

#include"../operation.h"

class ACTIVATION_FUNCTION : public OPERATION
{
public:
    void f(std::vector<std::shared_ptr<VARIABLE>>& inputs) override;
    std::shared_ptr<TENSOR<double>> bprop(std::vector<std::shared_ptr<VARIABLE>>& inputs, std::shared_ptr<VARIABLE> & focus, std::shared_ptr<TENSOR<double>> & gradient) override;

    virtual double activation_function(double x) = 0;
    virtual double activation_function_derivative(double x) = 0;
};

void ACTIVATION_FUNCTION::f(std::vector<std::shared_ptr<VARIABLE>>& inputs)
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("ACTIVATION_FUNCTION::f: Invalid number of input variables.");
    }

    std::shared_ptr<TENSOR<double>> _data = std::make_shared<TENSOR<double>>(inputs.front()->get_data()->shape());

    for (int i=0; i < _data->size(); i++) // apply activation function to all elements
    {
        _data->data()[i] = activation_function(inputs.front()->get_data()->data()[i]);
    }

    this->get_variable()->get_data() = _data;
}

std::shared_ptr<TENSOR<double>> ACTIVATION_FUNCTION::bprop(std::vector<std::shared_ptr<VARIABLE>>& inputs, std::shared_ptr<VARIABLE> & focus, std::shared_ptr<TENSOR<double>> & gradient)
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("ACTIVATION_FUNCTION::bprop: Invalid number of input variables.");
    }

    // load derivative of activation into data 
    std::shared_ptr<TENSOR<double>> _data = std::make_shared<TENSOR<double>>(focus->get_data()->shape());

    for (int i=0; i < _data->size(); i++) // apply derivative of activation function to all elements
    {
        _data->data()[i] = activation_function_derivative(focus->get_data()->data()[i]) * gradient->data()[i];
    }

    return _data;
}

#include "rectified_linear_unit.h"
#include "hyperbolic_tangent.h"
#include "linear.h"
#include "heavyside_step.h"
#include "sigmoid.h"

using ACTIVATION_FUNCTION_VARIANT = std::variant<ReLU, HyperbolicTangent, Linear, HeavysideStep, Sigmoid>; 


#endif // ACTIVATION_FUNCTION_INCLUDE_GUARD