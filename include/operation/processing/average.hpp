#ifndef AVERAGE_HPP
#define AVERAGE_HPP

#include "../operation.hpp"

/**
 * @brief the average operation is used to average the output of multiple Variables.
 */
class Average : public Operation
{
public:

    /**
     * @brief constructor for the average operation
     */
    Average() { mName = "AVERAGE"; };
    ~Average() = default;

    /**
     * @brief average the input tensors
     */
    void f(std::vector<std::shared_ptr<Variable>>& inputs) override;

    /**
     * @brief backward pass is not supported for average
     */
    std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient) override;
};

#endif // AVERAGE_HPP