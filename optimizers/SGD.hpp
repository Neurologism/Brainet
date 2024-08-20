#ifndef PRIMITIVESGD_HPP
#define PRIMITIVESGD_HPP

#include "optimizer.hpp"
#include "../module/module.hpp"

/**
 * @brief The SGD class represents the stochastic gradient descent optimizer.
 * @note Linearly decreases the learning rate from the initial learning rate to reach the final learning rate after a certain number of iterations.
 */
class SGD : public Optimizer
{
    double mInitialLearningRate; 
    double mFinalLearningRate;
    double mLastDecay;
    double mIteration = 0;

public:
    SGD(double initialLearningRate, double finalLearningRate, double lastDecay);
    SGD(double initialLearningRate, double lastDecay);
    ~SGD() = default;

    void update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;
};

SGD::SGD(double initialLearningRate, double finalLearningRate, double lastDecay) : mInitialLearningRate(initialLearningRate), mFinalLearningRate(finalLearningRate), mLastDecay(lastDecay)
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

SGD::SGD(double initialLearningRate, double lastDecay) : SGD(initialLearningRate, initialLearningRate / 100, lastDecay)
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

#endif // PRIMITIVESGD_HPP