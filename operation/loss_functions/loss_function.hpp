#ifndef LOSS_FUNCTION_HPP
#define LOSS_FUNCTION_HPP

#include "../operation.hpp"


/**
 * @brief the performance class is used to evaluate the performance of the model.
 */
class LossFunction : public Operation
{
public:
    /**
     * @brief constructor for the performance operation
     */
    LossFunction() { mName = "PERFORMANCE"; };
    ~LossFunction() = default;

    virtual void f(std::vector<std::shared_ptr<Variable>> &inputs) = 0;
    std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>> &inputs, std::shared_ptr<Variable> &focus, std::shared_ptr<Tensor<double>> &gradient) override;
};

std::shared_ptr<Tensor<double>> LossFunction::bprop(std::vector<std::shared_ptr<Variable>> &inputs, std::shared_ptr<Variable> &focus, std::shared_ptr<Tensor<double>> &gradient)
{
    return std::make_shared<Tensor<double>>(Tensor<double>(focus->getData()->shape(), 0)); // return a tensor with the same shape as the focus variable filled with zeros (this variable has no gradient)
}


// performance function variant
#include "error_rate.hpp"

using LossFunctionVariant = std::variant<ErrorRate>;


#endif // LOSS_FUNCTION_HPP