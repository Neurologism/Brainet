#ifndef PERFORMANCE_HPP
#define PERFORMANCE_HPP

#include "../operation.hpp"

/**
 * @brief the performance class is used to evaluate the performance of the model.
 */
class PerformanceFunction : public Operation
{
public:
    /**
     * @brief constructor for the performance operation
     */
    PerformanceFunction() { mName = "PERFORMANCE"; };
    ~PerformanceFunction() = default;

    /**
     * @brief backward pass is not supported for performance
     */
    std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>> &inputs, std::shared_ptr<Variable> &focus, std::shared_ptr<Tensor<double>> &gradient) override;
};

std::shared_ptr<Tensor<double>> PerformanceFunction::bprop(std::vector<std::shared_ptr<Variable>> &inputs, std::shared_ptr<Variable> &focus, std::shared_ptr<Tensor<double>> &gradient)
{
    throw std::runtime_error("PerformanceFunction: performance metrics should not be used for backpropagation");
}


// performance function variant

#include "error_rate.hpp"

using PerformanceFunctionVariant = std::variant<ErrorRate>;


#endif // PERFORMANCE_HPP