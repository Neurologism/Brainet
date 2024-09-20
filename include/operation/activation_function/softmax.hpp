#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "operation/operation.hpp"

/**
 * @brief Softmax function class, representing the softmax function f(x) = exp(x) / sum(exp(x)).
*/
class Softmax : public Operation
{   
    bool mUseWithExp = true;
protected:
    /**
     * @brief The softmax function.
     * @param inputs The input values.
    */
    void f(std::vector<std::shared_ptr<Variable>>& inputs) override;
    /**
     * @brief The derivative of the softmax function.
     * @param inputs The input values.
     * @param focus The focus tensor
     * @param gradient The gradient tensor
     * @return The gradient tensor
    */
    std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient) override;

public:
    Softmax() { mName = "SOFTMAX"; }
    ~Softmax() = default;

    void useWithLog();
};

#endif // SOFTMAX_HPP