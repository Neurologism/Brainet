#ifndef MEAN_ABSOLUTE_ERROR_HPP
#define MEAN_ABSOLUTE_ERROR_HPP

#include "../operation.hpp"

/**
 * @brief Mean absolute error class, representing the function f(x, y) = (1/n) * sum(|x_i - y_i|) for i = 1 to n.
 * @note The mean absolute error is a loss function that measures the average magnitude of the errors between the predicted values and the actual values.
 * @note The mean absolute error is less sensitive to outliers than the mean squared error.
 * @note We assume a gradient of 0 if the difference is 0.
 */
class MeanAbsoluteError : public Operation
{
public:
    MeanAbsoluteError() { mName = "MeanAbsoluteError"; }
    /**
     * @brief Calculates the mean absolute error.
     * @param inputs The input variables x and y.
    */
    void f(std::vector<std::shared_ptr<Variable>>& inputs) override;
    /**
     * @brief Calculates the gradient of the mean absolute error.
     * @param inputs The input variables x and y.
     * @param focus The focus variable.
     * @param gradient The gradient tensor.
     * @return The gradient tensor.
    */
    std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient) override;
};


void MeanAbsoluteError::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    // security checks
    if(inputs.size() != 2)
    {
        throw std::runtime_error("MeanAbsoluteError operation requires 2 inputs");
    }
    if(inputs[0]->getData()->shape() != inputs[1]->getData()->shape())
    {
        throw std::runtime_error("MeanAbsoluteError operation requires inputs to have the same shape");
    }
    if(inputs[0]->getData()->dimensionality() != 2)
    {
        throw std::runtime_error("MeanAbsoluteError::f: Other than 2D tensors are not supported");
    }

    // calculate the mean absolute error
    double sum = 0;
    for(std::uint32_t i = 0; i < inputs[0]->getData()->capacity(); i++)
    {
        sum += abs(inputs[0]->getData()->at(i) - inputs[1]->getData()->at(i));
    }
    sum /= inputs[0]->getData()->capacity();
    // store the result
    this->getVariable()->getData() = std::make_shared<Tensor<double>>(Tensor<double>({1},sum));
}

std::shared_ptr<Tensor<double>> MeanAbsoluteError::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient)
{
    // security checks
    if(inputs.size() != 2)
    {
        throw std::runtime_error("MeanAbsoluteError operation requires 2 inputs");
    }
    if(inputs[0]->getData()->shape() != inputs[1]->getData()->shape())
    {
        throw std::runtime_error("MeanAbsoluteError operation requires inputs to have the same shape");
    }
    if(inputs[0]->getData()->dimensionality() != 2)
    {
        throw std::runtime_error("MeanAbsoluteError::bprop: Other than 2D tensors are not supported");
    }

    // calculate the gradient
    std::shared_ptr<Tensor<double>> result = std::make_shared<Tensor<double>>(inputs[0]->getData()->shape());
    for(std::uint32_t i = 0; i < inputs[0]->getData()->capacity(); i++)
    {
        if(inputs[0]->getData()->at(i) > inputs[1]->getData()->at(i))
        {
            result->set(i, -gradient->at(0)/inputs[0]->getData()->capacity());
        }
        else if(inputs[0]->getData()->at(i) < inputs[1]->getData()->at(i))
        {
            result->set(i, gradient->at(0)/inputs[0]->getData()->capacity());
        }
        else
        {
            result->set(i, 0);
        }
    }
    return result;
}

#endif // MEAN_ABSOLUTE_ERROR_HPP