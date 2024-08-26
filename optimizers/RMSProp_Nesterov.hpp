#ifndef RMSPROP_NESTEROV_HPP
#define RMSPROP_NESTEROV_HPP

#include "optimizer.hpp"

/**
 * @brief The RMSPropNesterov class represents the RMSProp optimizer with Nesterov momentum.
 * RMSProp is an optimizer that utilizes the magnitude of recent gradients to normalize the gradients.
 */
class RMSPropNesterov : public Optimizer
{
    double mLearningRate;
    double mDecayRate;
    double mDelta;
    double mMomentum;
    std::vector<Tensor<double>> mCache;
    std::vector<Tensor<double>> mVelocity;

public:

    /**
     * @brief Constructs a new RMSPropNesterov object.
     * @param learningRate The learning rate.
     * @param decayRate The decay rate.
     * @param delta The delta.
     * @param momentum The momentum.
     * @param initialCache The initial cache.
     * @param initialVelocity The initial velocity.
     */
    RMSPropNesterov(double learningRate, double decayRate = 0.9, double delta = 1e-7, double momentum = 0.9, std::vector<Tensor<double>> initialCache = {}, std::vector<Tensor<double>> initialVelocity = {});

    ~RMSPropNesterov() = default;

    /**
     * @brief Initializes the optimizer.
     * @param rLearnableParameters The learnable parameters.
     */
    void __init__(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;

    /**
     * @brief Updates the learnable parameters using the RMSProp with Nesterov momentum algorithm.
     * @param rLearnableParameters The learnable parameters.
     */
    void update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;
};

RMSPropNesterov::RMSPropNesterov(double learningRate, double decayRate, double delta, double momentum, std::vector<Tensor<double>> initialCache, std::vector<Tensor<double>> initialVelocity) : mLearningRate(learningRate), mDecayRate(decayRate), mDelta(delta), mMomentum(momentum), mCache(initialCache), mVelocity(initialVelocity)
{
    if(mLearningRate <= 0 || mDecayRate <= 0 || mDecayRate >= 1 || mDelta <= 0 || mMomentum <= 0 || mMomentum >= 1)
    {
        throw std::invalid_argument("RMSPropNesterov::RMSPropNesterov: The learning rate, decay rate, and momentum must be positive and the decay rate and momentum must be less than 1");
    }
}

void RMSPropNesterov::__init__(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
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
        throw std::invalid_argument("RMSPropNesterov::__init__: The size of the cache must be equal to the number of learnable parameters");
    }

    if (mVelocity.empty())
    {
        for (const auto & rLearnableParameter : rLearnableParameters)
        {
            mVelocity.push_back(Tensor<double>(rLearnableParameter->getData()->shape(), 0.0));
        }
    }
    else if (mVelocity.size() != rLearnableParameters.size())
    {
        throw std::invalid_argument("RMSPropNesterov::__init__: The size of the velocity must be equal to the number of learnable parameters");
    }

    for ( std::size_t i = 0; i < rLearnableParameters.size(); i++)
    {
        for ( std::size_t j = 0; j < rLearnableParameters[i]->getData()->capacity(); j++)
        {
            rLearnableParameters[i]->getData()->add(j, mMomentum * mVelocity[i].at(j));
        }
    }
}

void RMSPropNesterov::update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    for ( std::size_t i = 0; i < rLearnableParameters.size(); i++)
    {
        std::shared_ptr<Tensor<double>> gradient = rLearnableParameters[i]->getData();

        for ( std::size_t j = 0; j < rLearnableParameters[i]->getData()->capacity(); j++)
        {
            rLearnableParameters[i]->getData()->subtract(j, mMomentum * mVelocity[i].at(j));
            mCache[i].set(j, mDecayRate * mCache[i].at(j) + (1 - mDecayRate) * gradient->at(j) * gradient->at(j));
            mVelocity[i].set(j, mMomentum * mVelocity[i].at(j) - mLearningRate * gradient->at(j) / (std::sqrt(mCache[i].at(j)) + mDelta));
            rLearnableParameters[i]->getData()->add(j, mVelocity[i].at(j) * (1 + mMomentum));
        }
    }
}


#endif // RMSPROP_NESTEROV_HPP