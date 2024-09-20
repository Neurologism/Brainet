#ifndef LOSS_HPP
#define LOSS_HPP

#include "../operation/loss_functions/loss_variant.hpp"
#include "../operation/surrogate_loss_functions/surrogate_loss_function.hpp"
#include "module.hpp"

/**
 * @brief the loss module is intended for calculating the loss as well as the surrogate loss of the model.
 */
class Loss final : public Module
{
    std::shared_ptr<Variable> mLossVariable; // storing the loss
    std::shared_ptr<Variable> mSurrogateLossVariable; // storing the surrogate loss

    void createVariables(const LossFunctionVariant &lossFunction, const SurrogateLossFunctionVariant &surrogateLossFunction);

public:
    /**
     * @brief add a loss function to the graph
     * @param lossFunction the operation representing the loss function.
     * @param name the name of the loss module
     * @note the surrogate loss function will be determined automatically
     */
    explicit Loss(const LossFunctionVariant& lossFunction, const std::string & name = "");

    /**
     * @brief add a loss function to the graph
     * @param lossFunction the operation representing the loss function.
     * @param surrogateLossFunction the operation representing the surrogate loss function.
     * @param name the name of the loss module
     */
    Loss(const LossFunctionVariant &lossFunction, const SurrogateLossFunctionVariant &surrogateLossFunction, const std::string & name = "");

    std::vector<std::shared_ptr<Variable>> getInputs() override;
    std::vector<std::shared_ptr<Variable>> getOutputs() override;
    std::vector<std::shared_ptr<Variable>> getLearnableVariables() override;
    std::vector<std::shared_ptr<Variable>> getGradientVariables() override;
};

#endif // LOSS_HPP