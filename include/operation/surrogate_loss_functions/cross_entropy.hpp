#ifndef CROSS_ENTROPY_HPP
#define CROSS_ENTROPY_HPP

#include "../operation.hpp"

/**
 * @brief The CrossEntropy class is a cost function that is used to train a model using the negative log likelyhood.
 */
class CrossEntropy : public Operation
{
    bool mUseWithLog = true;
public:
    /**
     * @brief constructor for the CrossEntropy operation
     */
    CrossEntropy() { mName = "NEGATIVE_LOG_LIKELYHOOD"; };
    ~CrossEntropy() = default;

    /**
     * @brief calculate the negative log likelyhood of the input tensors
     * @param inputs The input tensors
     * @note The first input tensor is the prediction and the second input tensor is the target.
     */
    void f(std::vector<std::shared_ptr<Variable>> &inputs) override;

    /**
     * @brief calculate the gradient of the negative log likelyhood
     * @param inputs The input tensors
     * @param focus The focus tensor
     * @param gradient The gradient tensor
     * @return The gradient tensor
     */
    std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>> &inputs, std::shared_ptr<Variable> &focus, std::shared_ptr<Tensor<double>> &gradient) override;

    void useWithExp();
};

#endif // CROSS_ENTROPY_HPP