#ifndef DROP_OUT_HPP
#define DROP_OUT_HPP

#include "../operation.hpp"

/**
 * @brief Dropout class, representing the dropout operation.
*/
class Dropout : public Operation
{
    double mDropoutRate;
    std::vector<bool> mMask;

public:
    Dropout(double dropoutRate) : mDropoutRate(dropoutRate) { mName = "DROPOUT"; }
    void f(std::vector<std::shared_ptr<Variable>>& inputs) override;
    std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient) override;
};

void Dropout::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    if(inputs.size() != 1)
    {
        throw std::runtime_error("Dropout: number of inputs is not 1");
    }

    auto input = inputs[0]->getData();
    auto result = std::make_shared<Tensor<double>>(Tensor<double>(input->shape()));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(mDropoutRate);

    mMask = std::vector<bool>(input->capacity());

    for(std::uint32_t i = 0; i < input->capacity(); i++)
    {
        mMask[i] = dist(gen);
        result->set({i}, input->at({i}) * mMask[i]);
    }

    this->getVariable()->getData() = result;
}

std::shared_ptr<Tensor<double>> Dropout::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient)
{
    if(inputs.size() != 1)
    {
        throw std::runtime_error("Dropout: number of inputs is not 1");
    }

    auto input = inputs[0]->getData();
    auto result = std::make_shared<Tensor<double>>(Tensor<double>(input->shape()));

    for(std::uint32_t i = 0; i < input->capacity(); i++)
    {
        result->set({i}, gradient->at({i}) * mMask[i]);
    }

    return result;
}

#endif // DROP_OUT_HPP