//
// Created by servant-of-scietia on 20.09.24.
//
#include "optimizer/sgd.hpp"

SGD::SGD(double initialLearningRate, double finalLearningRate, std::uint32_t lastDecay) : mInitialLearningRate(initialLearningRate), mFinalLearningRate(finalLearningRate), mLastDecay(lastDecay)
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

SGD::SGD(double initialLearningRate, std::uint32_t lastDecay) : SGD(initialLearningRate, initialLearningRate / 100, lastDecay)
{
}

void SGD::init(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{

}

void SGD::update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    double learningRate;
    if(mIteration > mLastDecay)
    {
        learningRate = mFinalLearningRate;
    }
    else
    {
        double decay = mIteration / mLastDecay;
        learningRate = (1 - decay) * mInitialLearningRate + decay * mFinalLearningRate;
    }
    for(std::uint32_t i = 0; i < rLearnableParameters.size(); i++)
    {
        for(std::uint32_t j = 0; j < rLearnableParameters[i]->getData()->capacity(); j++)
        {
            rLearnableParameters[i]->getData()->subtract(j, learningRate * GRAPH->getGradient(rLearnableParameters[i])->at(j));
        }
    }
    mIteration++;
}