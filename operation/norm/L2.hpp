#ifndef L2_HPP
#define L2_HPP

#include "norm.hpp"

/**
 * @brief Used to add a L2 norm penalty to a weight matrix. This is used to prevent overfitting.
 */
class L2 : public Norm
{
public:
    /**
     * @brief add a L2 norm penalty to the graph
     * @param lambda the lambda value to be used
     */
    L2(double lambda) : Norm(lambda) {__dbg_name = "L2";};
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
    std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient) override;
};

void L2::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("L2: number of inputs is not 1");
    }

    auto input = inputs[0]->get_data();
    auto result = std::make_shared<Tensor<double>>(Tensor<double>({1}));

    double sum = 0;
    for (std::uint32_t i = 0; i < input->capacity(); i++)
    {
        if ((i-1) % input->shape(0) == 0) // no penalty on the bias
        {
            continue;
        }
        sum += input->at({i}) * input->at({i});
    }

    result->set({0}, 0.5 * _lambda * sum);

    this->get_variable()->get_data() = result;
}


std::shared_ptr<Tensor<double>> L2::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient)
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("L2: number of inputs is not 1");
    }
    if (gradient->shape() != std::vector<size_t>({1}))
    {
        throw std::runtime_error("L2: gradient shape is not {1}");
    }

    auto input = inputs[0]->get_data();
    auto result = std::make_shared<Tensor<double>>(input->shape());

    for (std::uint32_t i = 0; i < input->capacity(); i++)
    {
        if ((i-1) % input->shape(0) == 0) // no penalty on the bias
        {
            result->set(i, 0);
        }
        else
        {
            result->set(i, _lambda * input->at(i));
        }
    }

    return result;
}

#endif // L2_HPP