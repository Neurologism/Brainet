#ifndef ENSEMBLE_MODULE_HPP
#define ENSEMBLE_MODULE_HPP

#include "module.hpp"
#include "cost.hpp"

/**
 * @brief the ensemble module is used to average the output of multiple Variables and apply a cost function to the output.
 */
class EnsembleModule : private Module
{
    std::shared_ptr<Variable> mOutputVariable; 
    std::shared_ptr<Variable> mCostVariable; 
    
    void __init__( std::vector<std::shared_ptr<Variable>> initialInpus, std::vector<std::shared_ptr<Variable>> initialOutputs ) override {};

public:
    /**
     * @brief constructor for the ensemble module
     * @param inputVariables the variables to average
     * @param costFunction the cost function to apply to the output
     */
    EnsembleModule(std::vector<std::shared_ptr<Variable>> inputVariables, CostVariant costFunction);

    ~EnsembleModule() = default;    

    /**
     * @brief function to get access to specific variables of the module.
     * @param index the index of the variable
     * @return the variable specified by the index
     * @note 0: output variable
     * @note 1: cost variable
     */
    std::shared_ptr<Variable> getVariable(std::uint32_t index) override;
};

EnsembleModule::EnsembleModule(std::vector<std::shared_ptr<Variable>> inputVariables, CostVariant costFunction)
{
    mOutputVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, inputVariables, {}))); // replace with average function

    mCostVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::visit([](auto&& arg) {
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, CostVariant{costFunction}), {mOutputVariable}, {})));

    // connections within the module
    mOutputVariable->getConsumers().push_back(mCostVariable);

    // other connections
    for(auto inputVariable : inputVariables)
    {
        inputVariable->getConsumers().push_back(mOutputVariable);
    }
}

std::shared_ptr<Variable> EnsembleModule::getVariable(std::uint32_t index)
{
    switch (index)
    {
    case 0:
        return mOutputVariable;
        break;
    case 1:
        return mCostVariable;
        break;
    default:
        throw std::invalid_argument("EnsembleModule::getVariable: index out of range");
        break;
    }
}

#endif // ENSEMBLE_MODULE_HPP
