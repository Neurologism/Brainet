#ifndef ACTIVATIONFUNCTION_INCLUDE_GUARD
#define ACTIVATIONFUNCTION_INCLUDE_GUARD

#include"../operation.h"

/**
 * @brief Base class for operation functions. Template class to create an activation function that executes elementwise.
 */
class ActivationFunction : public Operation
{
public:
    /**
     * @brief Construct a new ActivationFunction object
     */
    ActivationFunction() { __dbg_name = "ActivationFunction"; }
    /**
     * @brief Forward pass is similar for all activation functions. It applies the activation function to each element of the input tensor.
     */
    void f(std::vector<std::shared_ptr<Variable>>& inputs) override;
    /**
     * @brief Backward pass is similar for all activation functions. It applies the derivative of the activation function to each element of the gradient tensor.
     */
    std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient) override;

    /**
     * @brief Activation function to be implemented by the derived class.
     */
    virtual double activation_function(double x) = 0;
    /**
     * @brief Derivative of the activation function to be implemented by the derived class.
     */
    virtual double activation_function_derivative(double x) = 0;
};

void ActivationFunction::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    // always check for right number of inputs
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("ActivationFunction::f: Invalid number of input variables.");
    }

    std::shared_ptr<Tensor<double>> _data = std::make_shared<Tensor<double>>(inputs.front()->get_data()->shape()); // create a new tensor to store the result

    for (std::uint32_t i=0; i < _data->size(); i++) // apply activation function to all elements
    {
        _data->data()[i] = activation_function(inputs.front()->get_data()->data()[i]); // apply activation function
    }

    this->get_variable()->get_data() = _data; // store the result in the variable
}

std::shared_ptr<Tensor<double>> ActivationFunction::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient)
{
    // always check for right number of inputs
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("ActivationFunction::bprop: Invalid number of input variables.");
    }

    // load derivative of activation into data 
    std::shared_ptr<Tensor<double>> _data = std::make_shared<Tensor<double>>(focus->get_data()->shape());

    for (std::uint32_t i=0; i < _data->size(); i++) // apply derivative of activation function to all elements
    {
        _data->data()[i] = activation_function_derivative(focus->get_data()->data()[i]) * gradient->data()[i];
    }

    return _data;
}

// include all the activation functions to create an ActivationFunction_VARIANT

#include "rectified_linear_unit.h"
#include "hyperbolic_tangent.h"
#include "linear.h"
#include "heavyside_step.h"
#include "sigmoid.h"
#include "softmax.h"

using ActivationVariant = std::variant<ReLU, HyperbolicTangent, Linear, HeavysideStep, Sigmoid, Softmax>; // this can be used from the user side and should move to a different location at some point


#endif // ActivationFunction_INCLUDE_GUARD