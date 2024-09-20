//
// Created by servant-of-scietia on 20.09.24.
//
#include "optimizer/adagrad.hpp"

AdaGrad::AdaGrad(double learningRate, double delta, std::vector<Tensor<double>> initialSquaredGradients) : mLearningRate(learningRate), mDelta(delta), mSquaredGradients(initialSquaredGradients)
{
    if(mLearningRate <= 0 || mDelta <= 0)
    {
        throw std::invalid_argument("AdaGrad::AdaGrad: The learning rate and delta must be positive");
    }
}

void AdaGrad::__init__(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    if (mSquaredGradients.empty())
    {
        for (const auto & rLearnableParameter : rLearnableParameters)
        {
            mSquaredGradients.push_back(Tensor<double>(rLearnableParameter->getData()->shape(), 0.0));
        }
    }
    else if (mSquaredGradients.size() != rLearnableParameters.size())
    {
        throw std::invalid_argument("AdaGrad::__init__: The number of squared gradients must be equal to the number of learnable parameters");
    }
}

void AdaGrad::update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    for (std::size_t i = 0; i < rLearnableParameters.size(); i++)
    {
        std::shared_ptr<Tensor<double>> gradient = rLearnableParameters[i]->getData();

        for (std::size_t j = 0; j < gradient->capacity(); j++)
        {
            mSquaredGradients[i].add(j, gradient->at(j) * gradient->at(j));
            rLearnableParameters[i]->getData()->subtract(j, mLearningRate * gradient->at(j) / (std::sqrt(mSquaredGradients[i].at(j)) + mDelta));
        }
    }
}