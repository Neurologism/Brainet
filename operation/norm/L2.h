#ifndef L2_NORM_INCLUDE_GUARD
#define L2_NORM_INCLUDE_GUARD

#include "norm.h"

/**
 * @brief Used to add a L2 norm penalty to a weight matrix. This is used to prevent overfitting.
 */
class L2_NORM : public NORM
{
public:
    /**
     * @brief add a L2 norm penalty to the graph, using a default lambda value
     */
    L2_NORM() : NORM() {};
    /**
     * @brief add a L2 norm penalty to the graph
     * @param lambda the lambda value to be used
     */
    L2_NORM(double lambda) : NORM(lambda) {};
    /**
     * @brief compute the L2 norm of the input tensor
    */
    void f(std::vector<std::shared_ptr<VARIABLE>>& inputs) override;
    /**
     * @brief compute the gradient of the L2 norm penalty with respect to the input tensor
     * @param inputs the parents of the variable
     * @param focus this is the only variable everything else is constant
     * @param gradient the sum of the gradients of the consumers
    */
    std::shared_ptr<TENSOR<double>> bprop(std::vector<std::shared_ptr<VARIABLE>>& inputs, std::shared_ptr<VARIABLE> & focus, std::shared_ptr<TENSOR<double>> & gradient) override;
};

void L2_NORM::f(std::vector<std::shared_ptr<VARIABLE>>& inputs)
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("L2_NORM: number of inputs is not 1");
    }

    auto input = inputs[0]->get_data();
    auto result = std::make_shared<TENSOR<double>>(TENSOR<double>({1}));

    double sum = 0;
    for (std::uint32_t i = 0; i < input->size(); i++)
    {
        sum += input->at({i}) * input->at({i});
    }

    result->set({0}, 0.5 * _lambda * sum);

    this->get_variable()->get_data() = result;
}


std::shared_ptr<TENSOR<double>> L2_NORM::bprop(std::vector<std::shared_ptr<VARIABLE>>& inputs, std::shared_ptr<VARIABLE> & focus, std::shared_ptr<TENSOR<double>> & gradient)
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("L2_NORM: number of inputs is not 1");
    }

    auto input = inputs[0]->get_data();
    auto result = std::make_shared<TENSOR<double>>(input->shape());

    for (std::uint32_t i = 0; i < input->size(); i++)
    {
        result->data()[i] = -_lambda * input->data()[i] * gradient->data()[0];
    }

    return result;
}

#endif // L2_NORM_INCLUDE_GUARD