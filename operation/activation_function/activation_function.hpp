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
    ActivationFunction() { mName = "ActivationFunction"; }
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
    virtual double activationFunction(double x) = 0;
    /**
     * @brief Derivative of the activation function to be implemented by the derived class.
     */
    virtual double activationFunctionDerivative(double x) = 0;
};

void ActivationFunction::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    // always check for right number of inputs
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("ActivationFunction::f: Invalid number of input variables.");
    }

    std::shared_ptr<Tensor<double>> _data = std::make_shared<Tensor<double>>(inputs.front()->getData()->shape()); // create a new tensor to store the result

    for (std::uint32_t i=0; i < _data->capacity(); i++) // apply activation function to all elements
    {
        _data->set(i, activationFunction(inputs.front()->getData()->at(i))); // apply activation function
    }

    this->getVariable()->getData() = _data; // store the result in the variable
}

std::shared_ptr<Tensor<double>> ActivationFunction::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient)
{
    // always check for right number of inputs
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("ActivationFunction::bprop: Invalid number of input variables.");
    }

    // load derivative of activation into data 
    std::shared_ptr<Tensor<double>> _data = std::make_shared<Tensor<double>>(focus->getData()->shape());

    for (std::uint32_t i=0; i < _data->capacity(); i++) // apply derivative of activation function to all elements
    {
        _data->set(i, activationFunctionDerivative(inputs.front()->getData()->at(i))); // apply derivative of activation function
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

using HiddenVariant = std::variant<ReLU, HyperbolicTangent, Sigmoid>; // probably change later to one variant for all activation functions
using OutputVariant = std::variant<Linear, Sigmoid, Softmax>; 

#endif // ACTIVATIONFUNCTION_HPP