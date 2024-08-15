#ifndef PERFORMANCE_HPP
#define PERFORMANCE_HPP

#include "../operation.hpp"

/**
 * @brief the performance class is used to evaluate the performance of the model.
 */
class Performance : public Operation
{
public:
    /**
     * @brief constructor for the performance operation
     */
    Performance() { mName = "PERFORMANCE"; };
    ~Performance() = default;

    virtual void f(std::vector<std::shared_ptr<Variable>> &inputs) override;

    /**
     * @brief backward pass is not supported for performance
     */
    std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>> &inputs, std::shared_ptr<Variable> &focus, std::shared_ptr<Tensor<double>> &gradient) override;
};

std::shared_ptr<Tensor<double>> Performance::bprop(std::vector<std::shared_ptr<Variable>> &inputs, std::shared_ptr<Variable> &focus, std::shared_ptr<Tensor<double>> &gradient)
{
    throw std::runtime_error("Performance: performance metrics should not be used for backpropagation");
}

#endif // PERFORMANCE_HPP