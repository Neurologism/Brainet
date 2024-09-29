#ifndef MSE_HPP
#define MSE_HPP

#include"../operation.hpp"


/**
 * @brief Mean squared error class, representing the function f(x, y) = (1/n) * sum((x_i - y_i)^2) for i = 1 to n.
*/
class MSE : public Operation
{
public:
    MSE() { mName = "MSE"; }
    /**
     * @brief Calculates the mean squared error.
     * @param inputs The input variables x and y.
    */
    void f(std::vector<std::shared_ptr<Variable>>& inputs) override;
    /**
     * @brief Calculates the gradient of the mean squared error.
     * @param inputs The input variables x and y.
     * @param focus The focus variable.
     * @param gradient The gradient tensor.
     * @return The gradient tensor.
    */
    std::shared_ptr<Tensor> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor> & gradient) override;
};

#endif // MSE_HPP