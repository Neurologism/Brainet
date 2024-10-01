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
    bool mInitialized = false;
    std::vector<Tensor> mFirstMomentEstimates;
    std::vector<Tensor> mSecondMomentEstimates;
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
    Adam(double learningRate, double decayRate1 = 0.9, double decayRate2 = 0.999, double delta = 1e-8, std::vector<Tensor> initialFirstMomentEstimates = {}, std::vector<Tensor> initialSecondMomentEstimates = {});

    ~Adam() = default;

    /**
     * @brief Initializes the optimizer.
     * @param rLearnableParameters The learnable parameters.
     */
    void init(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters);

    /**
     * @brief Updates the learnable parameters using the Adam algorithm.
     * @param rLearnableParameters The learnable parameters.
     */
    void update(const std::vector<std::shared_ptr<Variable>> & rLearnableParameters) override;
};

#endif // ADAM_HPP

