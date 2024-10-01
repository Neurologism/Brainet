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
    std::shared_ptr<Tensor> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor> & gradient) override;
};

#endif // MEAN_ABSOLUTE_ERROR_HPP