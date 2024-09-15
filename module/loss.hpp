#ifndef LOSS_HPP
#define LOSS_HPP

#include "../operation/loss_functions/loss_function.hpp"
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

inline Loss::Loss(const LossFunctionVariant& lossFunction, const std::string & name) : Module(name)
{
    if (std::holds_alternative<ErrorRate>(lossFunction)) // Error Rate & Cross-Entropy
    {
        createVariables(lossFunction, CrossEntropy());
    }
    else
    {
        throw std::invalid_argument("Loss::Loss: the loss function has no default surrogate loss function");
    }
}

inline Loss::Loss(const LossFunctionVariant &lossFunction, const SurrogateLossFunctionVariant &surrogateLossFunction, const std::string & name) : Module(name)
{
    createVariables(lossFunction, surrogateLossFunction);
}

inline void Loss::createVariables(const LossFunctionVariant &lossFunction, const SurrogateLossFunctionVariant &surrogateLossFunction)
{
    // add variables to the graph

    mLossVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::visit([]<typename T0>(T0&& arg) {
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<T0>>(arg));}, LossFunctionVariant{lossFunction}))));

    mSurrogateLossVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::visit([]<typename T0>(T0&& arg) {
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<T0>>(arg));}, SurrogateLossFunctionVariant{surrogateLossFunction}))));
}

inline std::vector<std::shared_ptr<Variable>> Loss::getInputs()
{
    return {mLossVariable, mSurrogateLossVariable};
}

inline std::vector<std::shared_ptr<Variable>> Loss::getOutputs()
{
    return {mLossVariable, mSurrogateLossVariable};
}

inline std::vector<std::shared_ptr<Variable>> Loss::getLearnableVariables()
{
    return {};
}

inline std::vector<std::shared_ptr<Variable>> Loss::getGradientVariables()
{
    return {mSurrogateLossVariable};
}



#endif // LOSS_HPP