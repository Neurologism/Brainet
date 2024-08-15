#ifndef ERROR_RATE_HPP
#define ERROR_RATE_HPP

#include "performance.hpp"

/**
 * @brief the error rate operation is used to calculate the error rate of the model.
 */
class ErrorRate : public Performance
{
public:
    /**
     * @brief constructor for the error rate operation
     */
    ErrorRate() { mName = "ERROR_RATE"; };
    ~ErrorRate() = default;

    void f(std::vector<std::shared_ptr<Variable>> &inputs) override;
};

void ErrorRate::f(std::vector<std::shared_ptr<Variable>> &inputs)
{
    if (inputs.size() != 2)
    {
        throw std::runtime_error("ErrorRate: number of inputs is not 2");
    }

    if (inputs[0]->getData()->shape() != inputs[1]->getData()->shape())
    {
        throw std::runtime_error("ErrorRate: the shape of the inputs must be the same");
    }

    std::uint32_t size = inputs[0]->getData()->capacity();
    double error = 0;
    for (std::uint32_t i = 0; i < size; i++)
    {
        if (inputs[0]->getData()->at({i}) != inputs[1]->getData()->at({i}))
        {
            error++;
        }
    }

    this->getVariable()->getData() = std::make_shared<Tensor<double>>(Tensor<double>({1}));
    this->getVariable()->getData()->at({0}) = error / size;
}