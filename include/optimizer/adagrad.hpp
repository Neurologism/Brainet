#ifndef ADAGRAD_HPP
#define ADAGRAD_HPP

#include "optimizer.hpp"

/**
 * @brief The AdaGrad class represents the AdaGrad optimization algorithm.
 */
class AdaGrad : public Optimizer
{
    double mLearningRate;
    double mDelta;
    bool mInitialized = false;
    std::vector<Tensor> mSquaredGradients;

public:

    /**
     * @brief Constructs a new AdaGrad object.
     * @param learningRate The learning rate.
     * @param delta The delta value.
     * @param initialSquaredGradients The initial squared gradients.
     */
    AdaGrad(double learningRate, double delta = 1e-7, std::vector<Tensor> initialSquaredGradients = {});

    ~AdaGrad() = default;

    /**
     * @brief Initializes the optimizer.
     * @param rLearnableParameters The learnable parameters.
     */
    void init(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters);

    /**
     * @brief Updates the learnable parameters using the AdaGrad algorithm.
     * @param rLearnableParameters The learnable parameters.
     */
    void update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;
};

#endif // ADAGRAD_HPP