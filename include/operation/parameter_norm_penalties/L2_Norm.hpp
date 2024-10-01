#ifndef L2_NORM_HPP
#define L2_NORM_HPP

#include "parameter_norm_penalty.hpp"

/**
 * @brief Used to add a L2 norm penalty to a weight matrix. This is used to prevent overfitting.
 */
class L2Norm : public ParameterNormPenalty
{
public:
    /**
     * @brief add a L2 norm penalty to the graph
     * @param lambda the lambda value to be used
     */
    L2Norm(double lambda) : ParameterNormPenalty(lambda) { mName = "L2Norm"; };
    /**
     * @brief compute the L2 norm of the input tensor
    */
    void f(std::vector<std::shared_ptr<Variable>>& inputs) override;
    /**
     * @brief compute the gradient of the L2 norm penalty with respect to the input tensor
     * @param inputs the parents of the variable
     * @param focus this is the only variable everything else is constant
     * @param gradient the sum of the gradients of the consumers
    */
    std::shared_ptr<Tensor> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor> & gradient) override;
};

#endif // L2_NORM_HPP