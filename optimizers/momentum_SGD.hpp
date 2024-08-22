#ifndef MOMENTUM_SGD_HPP
#define MOMENTUM_SGD_HPP

#include "optimizer.hpp"

/**
 * @brief The Momentum class represents SGD with momentum. Momentum works by maintaining 
 * an exponentially decaying moving average of past gradients and thus targets variance in the stochastic gradient.
 */
class Momentum : public Optimizer
{
    std::vector<Tensor<double>> mVelocity; // velocity over time
    double mLearningRate;
    double mMomentum;

public:

    /**
     * @brief Constructs a new Momentum object.
     * @param learningRate The learning rate.
     * @param momentum The momentum.
     * @param initialVelocity The initial velocity.
     */
    Momentum(double learningRate, double momentum = 0.9, std::vector<Tensor<double>> initialVelocity = {});

    ~Momentum() = default;

    /**
     * @brief Initializes the optimizer.
     * @param rLearnableParameters The learnable parameters.
     */
    void __init__(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;

    /**
     * @brief Updates the learnable parameters using the momentum algorithm. 
     * @param rLearnableParameters The learnable parameters.
     */
    void update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;
};

Momentum::Momentum(double learningRate, double momentum, std::vector<Tensor<double>> initialVelocity) : mLearningRate(learningRate), mMomentum(momentum), mVelocity(initialVelocity)
{
    if(mLearningRate <= 0 || mMomentum <= 0 || mMomentum >= 1)
    {
        throw std::invalid_argument("Momentum::Momentum: The learning rate and momentum must be positive and the momentum must be less than 1");
    }

}

void Momentum::__init__(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
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
        throw std::invalid_argument("Momentum::__init__: The number of learnable parameters must be equal to the number of velocity tensors");
    }
}

void Momentum::update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    for (std::size_t i = 0; i < rLearnableParameters.size(); i++)
    {
        std::shared_ptr<Tensor<double>> gradient = GRAPH->getGradient(rLearnableParameters[i]);

        for (std::size_t j = 0; j < rLearnableParameters[i]->getData()->capacity(); j++)
        {
            mVelocity[i].set(j, mMomentum * mVelocity[i].at(j) - mLearningRate * gradient->at(j));
            rLearnableParameters[i]->getData()->add(j, mVelocity[i].at(j));
        }
    }
} 

#endif // MOMENTUM_SGD_HPP