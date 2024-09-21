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
    bool mInitialized = false;
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
    void init(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters);

    /**
     * @brief Updates the learnable parameters using the momentum algorithm. 
     * @param rLearnableParameters The learnable parameters.
     */
    void update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;
};

#endif // MOMENTUM_SGD_HPP