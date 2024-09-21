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
    bool mInitialized = false;
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
    void init(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters);

    /**
     * @brief Updates the learnable parameters using the Nesterov accelerated gradient algorithm. 
     * @param rLearnableParameters The learnable parameters.
     */
    void update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;
};

#endif // NESEROV_MOMENTUM_HPP