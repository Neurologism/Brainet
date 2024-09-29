//
// Created by servant-of-scietia on 20.09.24.
//
#include "optimizer/rmsprop.hpp"

RMSProp::RMSProp(double learningRate, double decayRate, double delta, std::vector<Tensor> initialCache) : mLearningRate(learningRate), mDecayRate(decayRate), mDelta(delta), mCache(initialCache)
{
    if(mLearningRate <= 0 || mDecayRate <= 0 || mDecayRate >= 1 || mDelta <= 0)
    {
        throw std::invalid_argument("RMSProp::RMSProp: The learning rate and decay rate must be positive and the decay rate must be less than 1");
    }
}

void RMSProp::init(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    if (mCache.empty())
    {
        for (const auto & rLearnableParameter : rLearnableParameters)
        {
            mCache.emplace_back(rLearnableParameter->getData()->shape(), 0.0);
        }
    }
    else if (mCache.size() != rLearnableParameters.size())
    {
        throw std::invalid_argument("RMSProp::init: The size of the cache must be equal to the number of learnable parameters");
    }
}

void RMSProp::update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    if (!mInitialized)
    {
        init(rLearnableParameters);
        mInitialized = true;
    }
    for ( std::size_t i = 0; i < rLearnableParameters.size(); i++)
    {
        std::shared_ptr<Tensor> gradient = GRAPH->getGradient(rLearnableParameters[i]);

        for ( std::size_t j = 0; j < gradient->capacity(); j++)
        {
            mCache[i].set(j, mDecayRate * mCache[i].at(j) + (1 - mDecayRate) * std::pow(gradient->at(j), 2));
            rLearnableParameters[i]->getData()->subtract(j, mLearningRate * gradient->at(j) / (std::sqrt(mCache[i].at(j)) + mDelta));
        }
    }
}