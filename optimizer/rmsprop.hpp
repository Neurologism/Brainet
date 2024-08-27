#ifndef RMSPROP_HPP
#define RMSPROP_HPP

#include "optimizer.hpp"

/**
 * @brief The RMSProp class represents the RMSProp optimizer.
 * RMSProp is an optimizer that utilizes the magnitude of recent gradients to normalize the gradients.
 */
class RMSProp : public Optimizer
{
    double mLearningRate;
    double mDecayRate;
    double mDelta;
    std::vector<Tensor<double>> mCache;

public:

    /**
     * @brief Constructs a new RMSProp object.
     * @param learningRate The learning rate.
     * @param decayRate The decay rate.
     * @param delta The delta.
     * @param initialCache The initial cache.
     */
    RMSProp(double learningRate, double decayRate = 0.9, double delta = 1e-7, std::vector<Tensor<double>> initialCache = {});

    ~RMSProp() = default;

    /**
     * @brief Initializes the optimizer.
     * @param rLearnableParameters The learnable parameters.
     */
    void __init__(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;

    /**
     * @brief Updates the learnable parameters using the RMSProp algorithm.
     * @param rLearnableParameters The learnable parameters.
     */
    void update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;
};

RMSProp::RMSProp(double learningRate, double decayRate, double delta, std::vector<Tensor<double>> initialCache) : mLearningRate(learningRate), mDecayRate(decayRate), mDelta(delta), mCache(initialCache)
{
    if(mLearningRate <= 0 || mDecayRate <= 0 || mDecayRate >= 1 || mDelta <= 0)
    {
        throw std::invalid_argument("RMSProp::RMSProp: The learning rate and decay rate must be positive and the decay rate must be less than 1");
    }
}

void RMSProp::__init__(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    if (mCache.empty())
    {
        for (const auto & rLearnableParameter : rLearnableParameters)
        {
            mCache.push_back(Tensor<double>(rLearnableParameter->getData()->shape(), 0.0));
        }
    }
    else if (mCache.size() != rLearnableParameters.size())
    {
        throw std::invalid_argument("RMSProp::__init__: The size of the cache must be equal to the number of learnable parameters");
    }
}

void RMSProp::update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    for ( std::size_t i = 0; i < rLearnableParameters.size(); i++)
    {
        std::shared_ptr<Tensor<double>> gradient = rLearnableParameters[i]->getData();

        for ( std::size_t j = 0; j < gradient->capacity(); j++)
        {
            mCache[i].set(j, mDecayRate * mCache[i].at(j) + (1 - mDecayRate) * std::pow(gradient->at(j), 2));
            rLearnableParameters[i]->getData()->subtract(j, mLearningRate * gradient->at(j) / (std::sqrt(mCache[i].at(j)) + mDelta));
        }
    }
}

#endif // RMSPROP_HPP