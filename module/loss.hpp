#ifndef COST_HPP
#define COST_HPP

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

public:
    /**
     * @brief add a loss function to the graph
     * @param lossFunction the operation representing the loss function.
     */
    Loss(LossFunctionVariant lossFunction);

    void remove();

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
     * @note 0: loss variable
     * @note 2: target variable
     */
    std::shared_ptr<Variable> getVariable(std::uint32_t index) override;
    
};

Loss::Loss(CostVariant lossFunction)
{
    // add variables to the graph
    mTargetVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {})));

    mCostVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::visit([](auto&& arg) {
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, CostVariant{lossFunction}), {mTargetVariable}, {})));
    
    // connections within the module
    mTargetVariable->getConsumers().push_back(mCostVariable);
}

void Loss::remove()
{
    // delete other connections
    for(auto input : mCostVariable->getInputs())
    {
        input->getConsumers().erase(std::find(input->getConsumers().begin(), input->getConsumers().end(), mCostVariable));
    }

    // delete the variables
    GRAPH->removeVariable(mTargetVariable);

    GRAPH->removeVariable(mCostVariable);
}


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

    mCostVariable->getInputs().push_back(initialInpus[0]);
}

std::shared_ptr<Variable> Loss::getVariable(std::uint32_t index)
{
    switch (index)
    {
    case 0:
        return mCostVariable;
        break;
    case 2:
        return mTargetVariable;
        break;
    default:
        throw std::invalid_argument("Loss::getVariable: index out of range");
        break;
    }
}



#endif // COST_HPP