#ifndef ADAGRAD_HPP
#define ADAGRAD_HPP

#include "Optimizer.hpp"

/**
 * @brief The AdaGrad class represents the AdaGrad optimization algorithm.
 */
class AdaGrad : public Optimizer
{
    double mLearningRate;
    double mDelta;
    std::vector<Tensor<double>> mSquaredGradients;

public:

    /**
     * @brief Constructs a new AdaGrad object.
     * @param learningRate The learning rate.
     * @param delta The delta value.
     * @param initialSquaredGradients The initial squared gradients.
     */
    AdaGrad(double learningRate, double delta = 1e-7, std::vector<Tensor<double>> initialSquaredGradients = {});

    ~AdaGrad() = default;

    /**
     * @brief Initializes the optimizer.
     * @param rLearnableParameters The learnable parameters.
     */
    void __init__(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;

    /**
     * @brief Updates the learnable parameters using the AdaGrad algorithm.
     * @param rLearnableParameters The learnable parameters.
     */
    void update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;
};

AdaGrad::AdaGrad(double learningRate, double delta, std::vector<Tensor<double>> initialSquaredGradients) : mLearningRate(learningRate), mDelta(delta), mSquaredGradients(initialSquaredGradients)
{
    if(mLearningRate <= 0 || mDelta <= 0)
    {
        throw std::invalid_argument("AdaGrad::AdaGrad: The learning rate and delta must be positive");
    }
}

void AdaGrad::__init__(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    if (mSquaredGradients.empty())
    {
        for (const auto & rLearnableParameter : rLearnableParameters)
        {
            mSquaredGradients.push_back(Tensor<double>(rLearnableParameter->getData()->shape(), 0.0));
        }
    }
    else if (mSquaredGradients.size() != rLearnableParameters.size())
    {
        throw std::invalid_argument("AdaGrad::__init__: The number of squared gradients must be equal to the number of learnable parameters");
    }
}

void AdaGrad::update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters)
{
    for (std::size_t i = 0; i < rLearnableParameters.size(); i++)
    {
        std::shared_ptr<Tensor<double>> gradient = rLearnableParameters[i]->getData();

        for (std::size_t j = 0; j < gradient->capacity(); j++)
        {
            mSquaredGradients[i].add(j, gradient->at(j) * gradient->at(j));
            rLearnableParameters[i]->getData()->subtract(j, mLearningRate * gradient->at(j) / (std::sqrt(mSquaredGradients[i].at(j)) + mDelta));
        }
    }
}

#endif // ADAGRAD_HPP