//
// Created by servant-of-scietia on 20.09.24.
//
#include "optimizer/nesterov_momentum.hpp"

NesterovMomentum::NesterovMomentum(double learningRate, double momentum, std::vector<Tensor> initialVelocity) : mLearningRate(learningRate), mMomentum(momentum), mVelocity(initialVelocity)
{
    if(mLearningRate <= 0 || mMomentum <= 0 || mMomentum >= 1)
    {
        throw std::invalid_argument("NesterovMomentum::NesterovMomentum: The learning rate and momentum must be positive and the momentum must be less than 1");
    }

}

void NesterovMomentum::init(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    if (mVelocity.empty())
    {
        for (const auto & rLearnableParameter : rLearnableParameters)
        {
            mVelocity.emplace_back(rLearnableParameter->getData()->shape(), 0.0);
        }
    }
    else if (mVelocity.size() != rLearnableParameters.size())
    {
        throw std::invalid_argument("NesterovMomentum::init: The size of the velocity vector must be equal to the size of the learnable parameters vector");
    }

    for ( std::size_t i = 0; i < rLearnableParameters.size(); i++)
    {
        for ( std::size_t j = 0; j < rLearnableParameters[i]->getData()->capacity(); j++)
        {
            rLearnableParameters[i]->getData()->add(j, mMomentum * mVelocity[i].at(j));
        }
    }
}

void NesterovMomentum::update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    if (!mInitialized)
    {
        init(rLearnableParameters);
        mInitialized = true;
    }
    for (std::size_t i = 0; i < rLearnableParameters.size(); i++)
    {
        std::shared_ptr<Tensor> gradient = GRAPH->getGradient(rLearnableParameters[i]);

        for (std::size_t j = 0; j < rLearnableParameters[i]->getData()->capacity(); j++)
        {
            rLearnableParameters[i]->getData()->subtract(j, mMomentum * mVelocity[i].at(j));
            mVelocity[i].set(j, mMomentum * mVelocity[i].at(j) - mLearningRate * gradient->at(j));
            rLearnableParameters[i]->getData()->add(j, mVelocity[i].at(j) * (1 + mMomentum));
        }
    }
}