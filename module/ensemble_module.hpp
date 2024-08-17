#ifndef ENSEMBLE_MODULE_HPP
#define ENSEMBLE_MODULE_HPP

#include "module.hpp"
#include "../operation/processing/average.hpp"
#include "../operation/cost_function/cost_function.hpp"


/**
 * @brief the ensemble module is used to average the output of multiple Variables and apply a cost function to the output.
 */
class EnsembleModule : private Module
{
    std::shared_ptr<Variable> mOutputVariable; 
    std::shared_ptr<Module> mCostModule; 
    
    // not supported
    void __init__( std::vector<std::shared_ptr<Variable>> initialInpus, std::vector<std::shared_ptr<Variable>> initialOutputs ) override {};

public:
    /**
     * @brief constructor for the ensemble module
     * @param inputVariables the variables to average
     * @param costFunction the cost function to apply to the output
     */
    EnsembleModule(std::vector<std::shared_ptr<Variable>> inputVariables, const Cost & costFunction);

    /**
     * @brief destructor for the ensemble module removes the module from the graph
     */
    void remove();

    /**
     * @brief function to get access to specific variables of the module.
     * @param index the index of the variable
     * @return the variable specified by the index
     * @note 0: output variable
     * @note 1: cost variable
     * @note 2: target variable
     */
    std::shared_ptr<Variable> getVariable(std::uint32_t index) override;
};

EnsembleModule::EnsembleModule(std::vector<std::shared_ptr<Variable>> inputVariables, const Cost & costFunction)
{
    mOutputVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Average>(), inputVariables, {})));

    mCostModule = std::make_shared<Cost>(costFunction);

    // connections within the module
    mOutputVariable->getConsumers().push_back(mCostModule->getVariable(0));
    mCostModule->__init__({mOutputVariable}, {});

    // other connections
    for(auto inputVariable : inputVariables)
    {
        inputVariable->getConsumers().push_back(mOutputVariable);
    }
}

void EnsembleModule::remove()
{
    // delete other connections
    for(auto input : mOutputVariable->getInputs())
    {
        input->getConsumers().erase(std::find(input->getConsumers().begin(), input->getConsumers().end(), mOutputVariable));
    }

    // delete the variables
    GRAPH->removeVariable(mOutputVariable);
}

std::shared_ptr<Variable> EnsembleModule::getVariable(std::uint32_t index)
{
    switch (index)
    {
    case 0:
        return mOutputVariable;
        break;
    case 1:
        return mCostModule->getVariable(1);
        break;
    case 2:
        return mCostModule->getVariable(2);
        break;
    default:
        throw std::invalid_argument("EnsembleModule::getVariable: index out of range");
        break;
    }
}

#endif // ENSEMBLE_MODULE_HPP
