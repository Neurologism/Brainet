//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/processing/dropout.hpp"

bool Dropout::msAveraging = false;

void Dropout::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    if(inputs.size() != 1)
    {
        throw std::runtime_error("Dropout: number of inputs is not 1");
    }

    auto input = inputs[0]->getData();
    auto result = std::make_shared<Tensor>(Tensor(input->shape()));

    if(msAveraging)
    {
        for(std::uint32_t i = 0; i < input->capacity(); i++)
        {
            result->set({i}, input->at({i}) * mDropoutRate);
        }
    }
    else
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dist(mDropoutRate);

        mMask = std::vector<bool>(input->capacity());

        for(std::uint32_t i = 0; i < input->capacity(); i++)
        {
            mMask[i] = dist(gen);
            result->set({i}, input->at({i}) * mMask[i]);
        }
    }
    this->getVariable()->getData() = result;
}

std::shared_ptr<Tensor> Dropout::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor> & gradient)
{
    if(inputs.size() != 1)
    {
        throw std::runtime_error("Dropout: number of inputs is not 1");
    }
    if(msAveraging)
    {
        throw std::runtime_error("Dropout: dropout is in averaging mode");
    }

    auto input = inputs[0]->getData();
    auto result = std::make_shared<Tensor>(Tensor(input->shape()));

    for(std::uint32_t i = 0; i < input->capacity(); i++)
    {
        result->set({i}, gradient->at({i}) * mMask[i]);
    }

    return result;
}

