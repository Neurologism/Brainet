//
// Created by servant-of-scietia on 20.09.24.
//
#include "optimizer/nesterov_momentum.hpp"

void NesterovMomentum::__init__(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
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
        throw std::invalid_argument("NesterovMomentum::__init__: The size of the velocity vector must be equal to the size of the learnable parameters vector");
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
    for (std::size_t i = 0; i < rLearnableParameters.size(); i++)
    {
        std::shared_ptr<Tensor<double>> gradient = GRAPH->getGradient(rLearnableParameters[i]);

        for (std::size_t j = 0; j < rLearnableParameters[i]->getData()->capacity(); j++)
        {
            rLearnableParameters[i]->getData()->subtract(j, mMomentum * mVelocity[i].at(j));
            mVelocity[i].set(j, mMomentum * mVelocity[i].at(j) - mLearningRate * gradient->at(j));
            rLearnableParameters[i]->getData()->add(j, mVelocity[i].at(j) * (1 + mMomentum));
        }
    }
}