//
// Created by servant-of-scietia on 20.09.24.
//
#include "optimizer/momentum_sgd.hpp"

Momentum::Momentum(double learningRate, double momentum, std::vector<Tensor<double>> initialVelocity) : mLearningRate(learningRate), mMomentum(momentum), mVelocity(initialVelocity)
{
    if(mLearningRate <= 0 || mMomentum <= 0 || mMomentum >= 1)
    {
        throw std::invalid_argument("Momentum::Momentum: The learning rate and momentum must be positive and the momentum must be less than 1");
    }

}

void Momentum::init(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    if (mVelocity.empty())
    {
        for (const auto & rLearnableParameter : rLearnableParameters)
        {
            mVelocity.push_back(Tensor<double>(rLearnableParameter->getData()->shape(), 0.0));
        }
    }
    else if (mVelocity.size() != rLearnableParameters.size())
    {
        throw std::invalid_argument("Momentum::init: The number of learnable parameters must be equal to the number of velocity tensors");
    }
}

void Momentum::update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    for (std::size_t i = 0; i < rLearnableParameters.size(); i++)
    {
        std::shared_ptr<Tensor<double>> gradient = GRAPH->getGradient(rLearnableParameters[i]);

        for (std::size_t j = 0; j < rLearnableParameters[i]->getData()->capacity(); j++)
        {
            mVelocity[i].set(j, mMomentum * mVelocity[i].at(j) - mLearningRate * gradient->at(j));
            rLearnableParameters[i]->getData()->add(j, mVelocity[i].at(j));
        }
    }
}