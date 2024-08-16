#ifndef PRIMITIVESGD_HPP
#define PRIMITIVESGD_HPP

#include "optimizer.hpp"
#include "../module/module.hpp"

/**
 * @brief The SGD class is a simple implementation of the stochastic gradient descent algorithm.
 */
class SGD : public Optimizer
{
    double mInitialLearningRate;
    double mDecayRate;

public:
    SGD(double initialLearningRate);
    ~SGD() = default;

    void update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters, std::uint32_t batchSize) override;
};

SGD::SGD(double initialLearningRate)
{
    mInitialLearningRate = initialLearningRate;
}

void SGD::update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters, std::uint32_t batchSize)
{
    for(std::uint32_t i = 0; i < rLearnableParameters.size(); i++)
    {
        for(std::uint32_t j = 0; j < rLearnableParameters[i]->getData()->capacity(); j++)
        {
            rLearnableParameters[i]->getData()->subtract(j, mInitialLearningRate * GRAPH->getGradient(rLearnableParameters[i])->at(j) / batchSize);
        }
    }
}

#endif // PRIMITIVESGD_HPP