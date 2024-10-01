//
// Created by servant-of-scietia on 20.09.24.
//
#include "optimizer/sgd.hpp"

SGD::SGD(Precision initialLearningRate, Precision finalLearningRate, std::uint32_t lastDecay) : mInitialLearningRate(initialLearningRate), mFinalLearningRate(finalLearningRate), mLastDecay(lastDecay)
{
    if(mInitialLearningRate <= 0 || mFinalLearningRate <= 0 || mLastDecay <= 0)
    {
        throw std::invalid_argument("SGD::SGD: The learning rates must be positive");
    }
    if(mInitialLearningRate <= mFinalLearningRate)
    {
        throw std::invalid_argument("SGD::SGD: The initial learning rate must be greater than the final learning rate");
    }
}

SGD::SGD(Precision initialLearningRate, std::uint32_t lastDecay) : SGD(initialLearningRate, initialLearningRate / 100, lastDecay)
{
}

void SGD::update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    Precision learningRate;
    if(mIteration > mLastDecay)
    {
        learningRate = mFinalLearningRate;
    }
    else
    {
        Precision decay = mIteration / mLastDecay;
        learningRate = (1 - decay) * mInitialLearningRate + decay * mFinalLearningRate;
    }
    for(const auto & rLearnableParameter : rLearnableParameters)
    {
        Tensor &parameterGradient = *GRAPH->getGradient(rLearnableParameter);
        Tensor &parameter = *rLearnableParameter->getData();

        for(std::uint64_t j = 0; j < parameter.capacity(); ++j)
        {
            parameter.subtract(j, learningRate * parameterGradient.at(j));
        }
    }
    mIteration++;
}