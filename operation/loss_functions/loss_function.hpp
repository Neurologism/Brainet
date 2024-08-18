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
    virtual std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>> &inputs, std::shared_ptr<Variable> &focus, std::shared_ptr<Tensor<double>> &gradient) = 0;
};


// performance function variant
#include "error_rate.hpp"

using LossFunctionVariant = std::variant<ErrorRate>;


#endif // LOSS_FUNCTION_HPP