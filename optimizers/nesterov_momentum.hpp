#ifndef NESEROV_MOMENTUM_HPP
#define NESEROV_MOMENTUM_HPP

#include "optimizer.hpp"

/**
 * @brief The NesterovMomentum class represents Nesterov accelerated gradient (NAG).
 * NAG works by first making a big jump in the direction of the previous accumulated gradient and then making a correction.
 */
class NesterovMomentum : public Optimizer
{
    std::vector<Tensor<double>> mVelocity; // velocity over time
    double mLearningRate;
    double mMomentum;

public:

    /**
     * @brief Constructs a new NesterovMomentum object.
     * @param learningRate The learning rate.
     * @param momentum The momentum.
     * @param initialVelocity The initial velocity.
     */
    NesterovMomentum(double learningRate, double momentum = 0.9, std::vector<Tensor<double>> initialVelocity = {});

    ~NesterovMomentum() = default;

    /**
     * @brief Initializes the optimizer.
     * @param rLearnableParameters The learnable parameters.
     */
    void __init__(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;

    /**
     * @brief Updates the learnable parameters using the Nesterov accelerated gradient algorithm. 
     * @param rLearnableParameters The learnable parameters.
     */
    void update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;
};

NesterovMomentum::NesterovMomentum(double learningRate, double momentum, std::vector<Tensor<double>> initialVelocity) : mLearningRate(learningRate), mMomentum(momentum), mVelocity(initialVelocity)
{
    if(mLearningRate <= 0 || mMomentum <= 0 || mMomentum >= 1)
    {
        throw std::invalid_argument("NesterovMomentum::NesterovMomentum: The learning rate and momentum must be positive and the momentum must be less than 1");
    }

}

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

            



#endif // NESEROV_MOMENTUM_HPP