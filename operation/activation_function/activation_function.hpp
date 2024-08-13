#ifndef ACTIVATIONFUNCTION_HPP
#define ACTIVATIONFUNCTION_HPP

#include"../operation.hpp"

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

    for (std::uint32_t i=0; i < _data->capacity(); i++) // apply activation function to all elements
    {
        _data->set(i, activation_function(inputs.front()->get_data()->at(i))); // apply activation function
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

    for (std::uint32_t i=0; i < _data->capacity(); i++) // apply derivative of activation function to all elements
    {
        _data->set(i, activation_function_derivative(inputs.front()->get_data()->at(i))); // apply derivative of activation function
    }

    return _data;
}

// include all the activation functions to create an ActivationFunction_VARIANT

#include "rectified_linear_unit.hpp"
#include "hyperbolic_tangent.hpp"
#include "linear.hpp"
#include "heavyside_step.hpp"
#include "sigmoid.hpp"
#include "softmax.hpp"

using ActivationVariant = std::variant<ReLU, HyperbolicTangent, Linear, HeavysideStep, Sigmoid, Softmax>; // this can be used from the user side and should move to a different location at some point


#endif // ACTIVATIONFUNCTION_HPP