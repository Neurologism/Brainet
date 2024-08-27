#ifndef ADAM_HPP
#define ADAM_HPP

#include "optimizer.hpp"

/**
 * @brief The Adam class represents the Adam optimizer.
 * Adam is an optimizer that computes adaptive learning rates for each parameter.
 */
class Adam : public Optimizer
{
    double mLearningRate;
    double mDecayRate1;
    double mDecayRate2;
    double mDelta;
    std::vector<Tensor<double>> mFirstMomentEstimates;
    std::vector<Tensor<double>> mSecondMomentEstimates;
    std::uint32_t mIteration = 0;

public:

    /**
     * @brief Constructs a new Adam object.
     * @param learningRate The learning rate.
     * @param decayRate1 The decay rate for the first moment estimates.
     * @param decayRate2 The decay rate for the second moment estimates.
     * @param delta The delta.
     * @param initialFirstMomentEstimates The initial first moment estimates.
     * @param initialSecondMomentEstimates The initial second moment estimates.
     */
    Adam(double learningRate, double decayRate1 = 0.9, double decayRate2 = 0.999, double delta = 1e-8, std::vector<Tensor<double>> initialFirstMomentEstimates = {}, std::vector<Tensor<double>> initialSecondMomentEstimates = {});

    ~Adam() = default;

    /**
     * @brief Initializes the optimizer.
     * @param rLearnableParameters The learnable parameters.
     */
    void __init__(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;

    /**
     * @brief Updates the learnable parameters using the Adam algorithm.
     * @param rLearnableParameters The learnable parameters.
     */
    void update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;
};

Adam::Adam(double learningRate, double decayRate1, double decayRate2, double delta, std::vector<Tensor<double>> initialFirstMomentEstimates, std::vector<Tensor<double>> initialSecondMomentEstimates) : mLearningRate(learningRate), mDecayRate1(decayRate1), mDecayRate2(decayRate2), mDelta(delta), mFirstMomentEstimates(initialFirstMomentEstimates), mSecondMomentEstimates(initialSecondMomentEstimates)
{
    if(mLearningRate <= 0 || mDecayRate1 <= 0 || mDecayRate1 >= 1 || mDecayRate2 <= 0 || mDecayRate2 >= 1 || mDelta <= 0)
    {
        throw std::invalid_argument("Adam::Adam: The learning rate, decay rates, and delta must be positive and the decay rates must be less than 1");
    }
}

void Adam::__init__(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    if (mFirstMomentEstimates.empty())
    {
        for (const auto & rLearnableParameter : rLearnableParameters)
        {
            mFirstMomentEstimates.push_back(Tensor<double>(rLearnableParameter->getData()->shape(), 0.0));
        }
    }
    else if (mFirstMomentEstimates.size() != rLearnableParameters.size())
    {
        throw std::invalid_argument("Adam::__init__: The number of first moment estimates must match the number of learnable parameters");
    }

    if (mSecondMomentEstimates.empty())
    {
        for (const auto & rLearnableParameter : rLearnableParameters)
        {
            mSecondMomentEstimates.push_back(Tensor<double>(rLearnableParameter->getData()->shape(), 0.0));
        }
    }
    else if (mSecondMomentEstimates.size() != rLearnableParameters.size())
    {
        throw std::invalid_argument("Adam::__init__: The number of second moment estimates must match the number of learnable parameters");
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

#endif // ADAM_HPP

