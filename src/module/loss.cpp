//
// Created by servant-of-scietia on 20.09.24.
//
#include "module/loss.hpp"

Loss::Loss(const LossFunctionVariant& lossFunction, const std::string & name) : Module(name)
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

Loss::Loss(const LossFunctionVariant &lossFunction, const SurrogateLossFunctionVariant &surrogateLossFunction, const std::string & name) : Module(name)
{
    createVariables(lossFunction, surrogateLossFunction);
}

void Loss::createVariables(const LossFunctionVariant &lossFunction, const SurrogateLossFunctionVariant &surrogateLossFunction)
{
    // add variables to the graph

    mLossVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::visit([]<typename T0>(T0&& arg) {
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<T0>>(arg));}, LossFunctionVariant{lossFunction}))));

    mSurrogateLossVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::visit([]<typename T0>(T0&& arg) {
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<T0>>(arg));}, SurrogateLossFunctionVariant{surrogateLossFunction}))));
}

std::vector<std::shared_ptr<Variable>> Loss::getInputs()
{
    return {mLossVariable, mSurrogateLossVariable};
}

std::vector<std::shared_ptr<Variable>> Loss::getOutputs()
{
    return {mLossVariable, mSurrogateLossVariable};
}

std::vector<std::shared_ptr<Variable>> Loss::getLearnableVariables()
{
    return {};
}

std::vector<std::shared_ptr<Variable>> Loss::getGradientVariables()
{
    return {mSurrogateLossVariable};
}