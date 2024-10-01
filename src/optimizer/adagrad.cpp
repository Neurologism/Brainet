//
// Created by servant-of-scietia on 20.09.24.
//
#include "optimizer/adagrad.hpp"

AdaGrad::AdaGrad(double learningRate, double delta, std::vector<Tensor> initialSquaredGradients) : mLearningRate(learningRate), mDelta(delta), mSquaredGradients(initialSquaredGradients)
{
    if(mLearningRate <= 0 || mDelta <= 0)
    {
        throw std::invalid_argument("AdaGrad::AdaGrad: The learning rate and delta must be positive");
    }
}

void AdaGrad::init(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    if (mSquaredGradients.empty())
    {
        for (const auto & rLearnableParameter : rLearnableParameters)
        {
            mSquaredGradients.emplace_back(rLearnableParameter->getData()->shape(), 0.0);
        }
    }
    else if (mSquaredGradients.size() != rLearnableParameters.size())
    {
        throw std::invalid_argument("AdaGrad::init: The number of squared gradients must be equal to the number of learnable parameters");
    }
}

void AdaGrad::update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    if (!mInitialized)
    {
        init(rLearnableParameters);
        mInitialized = true;
    }
    for (std::size_t i = 0; i < rLearnableParameters.size(); i++)
    {
        std::shared_ptr<Tensor> gradient = GRAPH->getGradient(rLearnableParameters[i]);

        for (std::size_t j = 0; j < gradient->capacity(); j++)
        {
            mSquaredGradients[i].add(j, gradient->at(j) * gradient->at(j));
            rLearnableParameters[i]->getData()->subtract(j, mLearningRate * gradient->at(j) / (std::sqrt(mSquaredGradients[i].at(j)) + mDelta));
        }
    }
}