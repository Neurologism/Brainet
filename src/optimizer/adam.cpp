//
// Created by servant-of-scietia on 20.09.24.
//
#include "optimizer/adam.hpp"

Adam::Adam(double learningRate, double decayRate1, double decayRate2, double delta, std::vector<Tensor<double>> initialFirstMomentEstimates, std::vector<Tensor<double>> initialSecondMomentEstimates) : mLearningRate(learningRate), mDecayRate1(decayRate1), mDecayRate2(decayRate2), mDelta(delta), mFirstMomentEstimates(initialFirstMomentEstimates), mSecondMomentEstimates(initialSecondMomentEstimates)
{
    if(mLearningRate <= 0 || mDecayRate1 <= 0 || mDecayRate1 >= 1 || mDecayRate2 <= 0 || mDecayRate2 >= 1 || mDelta <= 0)
    {
        throw std::invalid_argument("Adam::Adam: The learning rate, decay rates, and delta must be positive and the decay rates must be less than 1");
    }
}

void Adam::init(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    if (mFirstMomentEstimates.empty())
    {
        for (const auto & rLearnableParameter : rLearnableParameters)
        {
            mFirstMomentEstimates.emplace_back(rLearnableParameter->getData()->shape(), 0.0);
        }
    }
    else if (mFirstMomentEstimates.size() != rLearnableParameters.size())
    {
        throw std::invalid_argument("Adam::init: The number of first moment estimates must match the number of learnable parameters");
    }

    if (mSecondMomentEstimates.empty())
    {
        for (const auto & rLearnableParameter : rLearnableParameters)
        {
            mSecondMomentEstimates.emplace_back(rLearnableParameter->getData()->shape(), 0.0);
        }
    }
    else if (mSecondMomentEstimates.size() != rLearnableParameters.size())
    {
        throw std::invalid_argument("Adam::init: The number of second moment estimates must match the number of learnable parameters");
    }
}

void Adam::update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    mIteration++;
    for (std::size_t i = 0; i < rLearnableParameters.size(); i++)
    {
        std::shared_ptr<Tensor<double>> gradient = rLearnableParameters[i]->getData();
        for (std::size_t j = 0; j < gradient->capacity(); j++)
        {
            mFirstMomentEstimates[i].set(j, mDecayRate1 * mFirstMomentEstimates[i].at(j) + (1 - mDecayRate1) * gradient->at(j));
            mSecondMomentEstimates[i].set(j, mDecayRate2 * mSecondMomentEstimates[i].at(j) + (1 - mDecayRate2) * gradient->at(j) * gradient->at(j));
            double firstMomentEstimateBiasCorrected = mFirstMomentEstimates[i].at(j) / (1 - std::pow(mDecayRate1, mIteration));
            double secondMomentEstimateBiasCorrected = mSecondMomentEstimates[i].at(j) / (1 - std::pow(mDecayRate2, mIteration));
            rLearnableParameters[i]->getData()->subtract(j, mLearningRate * firstMomentEstimateBiasCorrected / (std::sqrt(secondMomentEstimateBiasCorrected) + mDelta));
        }
    }
}