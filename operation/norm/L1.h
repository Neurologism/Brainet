#ifndef L1_NORM_INCLUDE_GUARD
#define L1_NORM_INCLUDE_GUARD

#include "norm.h"

/**
 * @brief Used to add a L1 norm penalty to a weight matrix. This is used to prevent overfitting.
 */
class L1_NORM : public NORM
{
public:
    /**
    * @brief add a L1 norm penalty to the graph
    * @param lambda the lambda value to be used
    */
    L1_NORM(double lambda) : NORM(lambda) {};
    ~L1_NORM() = default;
    /**
    * @brief compute the L1 norm of the input tensor
    */
    void f(std::vector<std::shared_ptr<VARIABLE>>& inputs) override;
    /**
    * @brief compute the gradient of the L1 norm penalty with respect to the input tensor
    * @param inputs the parents of the variable
    * @param focus this is the only variable everything else is constant
    * @param gradient the sum of the gradients of the consumers
    */
    std::shared_ptr<TENSOR<double>> bprop(std::vector<std::shared_ptr<VARIABLE>>& inputs, std::shared_ptr<VARIABLE> & focus, std::shared_ptr<TENSOR<double>> & gradient) override;
};

void L1_NORM::f(std::vector<std::shared_ptr<VARIABLE>>& inputs)
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("L1_NORM: number of inputs is not 1");
    }

    auto input = inputs[0]->get_data();
    auto result = std::make_shared<TENSOR<double>>(TENSOR<double>({1}));

    double sum = 0;
    for (std::uint32_t i = 0; i < input->size(); i++)
    {
        sum += std::abs(input->at({i}));
    }

    result->set({0}, _lambda * sum);

    this->get_variable()->get_data() = result;
}

std::shared_ptr<TENSOR<double>> L1_NORM::bprop(std::vector<std::shared_ptr<VARIABLE>>& inputs, std::shared_ptr<VARIABLE> & focus, std::shared_ptr<TENSOR<double>> & gradient)
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("L1_NORM: number of inputs is not 1");
    }

    auto input = inputs[0]->get_data();
    auto result = std::make_shared<TENSOR<double>>(TENSOR<double>(input->shape()));

    for (std::uint32_t i = 0; i < input->size(); i++)
    {
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

#endif // L1_NORM_INCLUDE_GUARD