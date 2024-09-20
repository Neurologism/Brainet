#ifndef SGD_HPP
#define SGD_HPP

#include "optimizer.hpp"
#include "../module/module.hpp"

/**
 * @brief The SGD class represents the stochastic gradient descent optimizer.
 * @note Linearly decreases the learning rate from the initial learning rate to reach the final learning rate after a certain number of iterations.
 */
class SGD : public Optimizer
{
    double mInitialLearningRate; 
    double mFinalLearningRate;
    double mLastDecay;
    double mIteration = 0;

public:
    /**
     * @brief Constructs a new SGD object.
     * @param initialLearningRate The initial learning rate.
     * @param finalLearningRate The final learning rate.
     * @param lastDecay The number of iterations after which the learning rate will be the final learning rate.
     */
    SGD(double initialLearningRate, double finalLearningRate, std::uint32_t lastDecay);
    /**
     * @brief Constructs a new SGD object.
     * @param initialLearningRate The initial learning rate.
     * @param lastDecay The number of iterations after which the learning rate will be the final learning rate.
     */
    SGD(double initialLearningRate, std::uint32_t lastDecay);
    ~SGD() = default;

    /**
     * @brief Initializes the optimizer.
     * @param rLearnableParameters The learnable parameters.
     */
    void init(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;

    /**
     * @brief Updates the learnable parameters using the stochastic gradient descent algorithm.
     * @param rLearnableParameters The learnable parameters.
     */
    void update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;
};

#endif //SGD_HPP