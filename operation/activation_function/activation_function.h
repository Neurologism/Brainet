#ifndef ACTIVATION_FUNCTION_INCLUDE_GUARD
#define ACTIVATION_FUNCTION_INCLUDE_GUARD

#include"../operation.h"

/**
 * @brief Base class for operation functions. Derive from this class to create an activation function.
 */
class ACTIVATION_FUNCTION : public OPERATION
{
public:
    /**
     * @brief Forward pass is similar for all activation functions. It applies the activation function to each element of the input tensor.
     */
    void f(std::vector<std::shared_ptr<VARIABLE>>& inputs) override;
    /**
     * @brief Backward pass is similar for all activation functions. It applies the derivative of the activation function to each element of the gradient tensor.
     */
    std::shared_ptr<TENSOR<double>> bprop(std::vector<std::shared_ptr<VARIABLE>>& inputs, std::shared_ptr<VARIABLE> & focus, std::shared_ptr<TENSOR<double>> & gradient) override;

    /**
     * @brief Activation function to be implemented by the derived class.
     */
    virtual double activation_function(double x) = 0;
    /**
     * @brief Derivative of the activation function to be implemented by the derived class.
     */
    virtual double activation_function_derivative(double x) = 0;
};

void ACTIVATION_FUNCTION::f(std::vector<std::shared_ptr<VARIABLE>>& inputs)
{
    // always check for right number of inputs
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("ACTIVATION_FUNCTION::f: Invalid number of input variables.");
    }

    std::shared_ptr<TENSOR<double>> _data = std::make_shared<TENSOR<double>>(inputs.front()->get_data()->shape()); // create a new tensor to store the result

    for (int i=0; i < _data->size(); i++) // apply activation function to all elements
    {
        _data->data()[i] = activation_function(inputs.front()->get_data()->data()[i]); // apply activation function
    }

    this->get_variable()->get_data() = _data; // store the result in the variable
}

std::shared_ptr<TENSOR<double>> ACTIVATION_FUNCTION::bprop(std::vector<std::shared_ptr<VARIABLE>>& inputs, std::shared_ptr<VARIABLE> & focus, std::shared_ptr<TENSOR<double>> & gradient)
{
    // always check for right number of inputs
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

// include all the activation functions to create an ACTIVATION_FUNCTION_VARIANT

#include "rectified_linear_unit.h"
#include "hyperbolic_tangent.h"
#include "linear.h"
#include "heavyside_step.h"
#include "sigmoid.h"

using ACTIVATION_FUNCTION_VARIANT = std::variant<ReLU, HyperbolicTangent, Linear, HeavysideStep, Sigmoid>; // this can be used from the user side and should move to a different location at some point


#endif // ACTIVATION_FUNCTION_INCLUDE_GUARD