#ifndef LOSS_FUNCTION_HPP
#define LOSS_FUNCTION_HPP

#include "operation/operation.hpp"


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
    std::shared_ptr<Tensor> bprop(std::vector<std::shared_ptr<Variable>> &inputs, std::shared_ptr<Variable> &focus, std::shared_ptr<Tensor> &gradient) override;
};


#endif // LOSS_FUNCTION_HPP