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

void Average::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    if (inputs.size() < 2)
    {
        throw std::runtime_error("Average: number of inputs is less than 2");
    }

    std::uint32_t size = inputs[0]->getData()->capacity();
    for (std::uint32_t i = 1; i < inputs.size(); i++)
    {
        if (inputs[i]->getData()->capacity() != size)
        {
            throw std::runtime_error("Average: the size of all inputs must be the same");
        }
    }

    auto result = std::make_shared<Tensor<double>>(Tensor<double>(inputs[0]->getData()->shape()));

    for (std::uint32_t i = 0; i < size; i++)
    {
        double sum = 0;
        for (std::uint32_t j = 0; j < inputs.size(); j++)
        {
            sum += inputs[j]->getData()->at({i});
        }
        result->set({i}, sum / inputs.size());
    }

    this->getVariable()->getData() = result;
}

std::shared_ptr<Tensor<double>> Average::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient)
{
    throw std::runtime_error("Average: backward pass is currently not supported");
}

#endif // AVERAGE_HPP