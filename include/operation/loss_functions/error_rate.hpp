#ifndef ERROR_RATE_HPP
#define ERROR_RATE_HPP

#include "loss_function.hpp" 

/**
 * @brief the error rate operation is used to calculate the error rate of the model.
 */
class ErrorRate : public LossFunction
{
public:
    /**
     * @brief constructor for the error rate operation
     */
    ErrorRate() { mName = "ERROR_RATE"; };
    ~ErrorRate() = default;

    /**
     * @brief calculate the error rate of the input tensors
     * @param inputs The input tensors
     * @note The first input tensor is the prediction and the second input tensor is the target.
     */
    void f(std::vector<std::shared_ptr<Variable>> &inputs) override;
};

#endif // ERROR_RATE_HPP