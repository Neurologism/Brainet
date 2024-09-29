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
    bool mInitialized = false;
    std::vector<Tensor> mCache;

public:

    /**
     * @brief Constructs a new RMSProp object.
     * @param learningRate The learning rate.
     * @param decayRate The decay rate.
     * @param delta The delta.
     * @param initialCache The initial cache.
     */
    RMSProp(double learningRate, double decayRate = 0.9, double delta = 1e-7, std::vector<Tensor> initialCache = {});

    ~RMSProp() = default;

    /**
     * @brief Initializes the optimizer.
     * @param rLearnableParameters The learnable parameters.
     */
    void init(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters);

    /**
     * @brief Updates the learnable parameters using the RMSProp algorithm.
     * @param rLearnableParameters The learnable parameters.
     */
    void update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;
};

#endif // RMSPROP_HPP