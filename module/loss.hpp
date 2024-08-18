#ifndef LOSS_HPP
#define LOSS_HPP

#include "../dependencies.hpp"
#include "module.hpp"
#include "../operation/loss_functions/loss_function.hpp"
#include "../operation/surrogate_loss_functions/surrogate_loss_function.hpp"
#include "../operation/processing/one_hot.hpp"

/**
 * @brief the loss module is intended for calculating the loss as well as the surrogate loss of the model.
 
 */
class Loss : public Module
{
    std::shared_ptr<Variable> mTargetVariable; // storing the target
    std::shared_ptr<Variable> mLossVariable; // storing the loss
    std::shared_ptr<Variable> mSurrogateLossVariable; // storing the surrogate loss

    void createVariables(LossFunctionVariant lossFunction, SurrogateLossFunctionVariant surrogateLossFunction);

public:
    /**
     * @brief add a loss function to the graph
     * @param lossFunction the operation representing the loss function.
     * @note the surrogate loss function will be determined automatically
     */
    Loss(LossFunctionVariant lossFunction);

    /**
     * @brief add a loss function to the graph
     * @param lossFunction the operation representing the loss function.
     * @param surrogateLossFunction the operation representing the surrogate loss function.
     */
    Loss(LossFunctionVariant lossFunction, SurrogateLossFunctionVariant surrogateLossFunction);

    // void remove();

    /**
     * @brief used to initialize the module with the input and output variables.
     * @param initialInpus the input variables
     * @param initialOutputs the output variables
     */
    void __init__( std::vector<std::shared_ptr<Variable>> initialInpus, std::vector<std::shared_ptr<Variable>> initialOutputs ) override;

    /**
     * @brief function to get access to specific variables of the module.
     * @param index the index of the variable
     * @return the variable specified by the index
     * @note 0: surrogate loss variable
     * @note 1: loss variable
     * @note 2: target variable
     */
    std::shared_ptr<Variable> getVariable(std::uint32_t index) override;
    
};

Loss::Loss(LossFunctionVariant lossFunction)
{
    if (std::holds_alternative<ErrorRate>(lossFunction)) // Error Rate & Cross Entropy
    {
        createVariables(lossFunction, CrossEntropy());
    }
    else
    {
        throw std::invalid_argument("Loss::Loss: the loss function has no default surrogate loss function");
    }
}

Loss::Loss(LossFunctionVariant lossFunction, SurrogateLossFunctionVariant surrogateLossFunction)
{
    createVariables(lossFunction, surrogateLossFunction);
}

void Loss::createVariables(LossFunctionVariant lossFunction, SurrogateLossFunctionVariant surrogateLossFunction)
{
    // add variables to the graph
    mTargetVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {})));

    mLossVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::visit([](auto&& arg) {
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, LossFunctionVariant{lossFunction}), {mTargetVariable}, {})));

    mSurrogateLossVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::visit([](auto&& arg) {
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, SurrogateLossFunctionVariant{surrogateLossFunction}), {mTargetVariable}, {})));
    
    // connections within the module
    mTargetVariable->getConsumers().push_back(mLossVariable);
    mTargetVariable->getConsumers().push_back(mSurrogateLossVariable);
}

// void Loss::remove()
// {
//     // delete other connections
//     for(auto input : mCostVariable->getInputs())
//     {
//         input->getConsumers().erase(std::find(input->getConsumers().begin(), input->getConsumers().end(), mCostVariable));
//     }

//     // delete the variables
//     GRAPH->removeVariable(mTargetVariable);

//     GRAPH->removeVariable(mCostVariable);
// }


void Loss::__init__( std::vector<std::shared_ptr<Variable>> initialInpus, std::vector<std::shared_ptr<Variable>> initialOutputs )
{
    if (initialInpus.size() != 1)
    {
        throw std::invalid_argument("Loss::__init__: the number of input variables must be 1");
    }
    if (initialOutputs.size() != 0)
    {
        throw std::invalid_argument("Loss::__init__: the number of output variables must be 0");
    }

    mLossVariable->getInputs().push_back(initialInpus[0]);
    mSurrogateLossVariable->getInputs().push_back(initialInpus[0]);
}

std::shared_ptr<Variable> Loss::getVariable(std::uint32_t index)
{
    switch (index)
    {
    case 0:
        return mSurrogateLossVariable;
        break;
    case 1:
        return mLossVariable;
        break;
    case 2:
        return mTargetVariable;
        break;
    default:
        throw std::invalid_argument("Loss::getVariable: index out of range");
        break;
    }
}



#endif // LOSS_HPP