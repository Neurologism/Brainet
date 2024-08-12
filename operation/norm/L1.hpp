#ifndef L1_HPP
#define L1_HPP

#include "norm.hpp"

/**
 * @brief Used to add a L1 norm penalty to a weight matrix. This is used to prevent overfitting.
 */
class L1 : public Norm
{
public:
    /**
    * @brief add a L1 norm penalty to the graph
    * @param lambda the lambda value to be used
    */
    L1(double lambda) : Norm(lambda) { __dbg_name = "L1"; };
    ~L1() = default;
    /**
    * @brief compute the L1 norm of the input tensor
    */
    void f(std::vector<std::shared_ptr<Variable>>& inputs) override;
    /**
    * @brief compute the gradient of the L1 norm penalty with respect to the input tensor
    * @param inputs the parents of the variable
    * @param focus this is the only variable everything else is constant
    * @param gradient the sum of the gradients of the consumers
    */
    std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient) override;
};

void L1::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("L1: number of inputs is not 1");
    }

    auto input = inputs[0]->get_data();
    auto result = std::make_shared<Tensor<double>>(Tensor<double>({1}));

    double sum = 0;
    for (std::uint32_t i = 0; i < input->size(); i++)
    {
        if ((i - 1) % input->shape(0) == 0) // no penalty on the bias
        {
            continue;
        }
        sum += std::abs(input->at({i}));
    }

    result->set({0}, _lambda * sum);

    this->get_variable()->get_data() = result;
}

std::shared_ptr<Tensor<double>> L1::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient)
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("L1: number of inputs is not 1");
    }

    auto input = inputs[0]->get_data();
    auto result = std::make_shared<Tensor<double>>(Tensor<double>(input->shape()));

    for (std::uint32_t i = 0; i < input->size(); i++)
    {
        if ((i - 1) % input->shape(0) == 0) // no penalty on the bias
        {
            result->set({i}, 0);
            continue;
        }

        if (input->at({i}) > 0)
        {
            result->set({i}, _lambda);
        }
        else if (input->at({i}) < 0)
        {
            result->set({i}, -_lambda);
        }
        else
        {
            result->set({i}, 0);
        }
    }

    return result;
}

#endif // L1_HPP