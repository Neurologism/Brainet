#ifndef ENSEMBLE_MODULE_HPP
#define ENSEMBLE_MODULE_HPP

#include "module.hpp"
#include "../operation/processing/average.hpp"


/**
 * @brief the ensemble module is used to average the output of multiple Variables and apply a cost function to the output.
 */
class EnsembleModule : private Module
{
    std::shared_ptr<Variable> mOutputVariable; 
    std::shared_ptr<Module> mLossModule; 
    
    // not supported
    void __init__( std::vector<std::shared_ptr<Variable>> initialInpus, std::vector<std::shared_ptr<Variable>> initialOutputs ) override {};

public:
    /**
     * @brief constructor for the ensemble module
     * @param inputVariables the variables to average
     * @param lossModule the loss module to apply to the output
     */
    EnsembleModule(std::vector<std::shared_ptr<Variable>> inputVariables, const Loss & lossModule);

    // /**
    //  * @brief destructor for the ensemble module removes the module from the graph
    //  */
    // void remove();

    /**
     * @brief function to get access to specific variables of the module.
     * @param index the index of the variable
     * @return the variable specified by the index
     * @note 0: output variable
     * @note 1: surrogate loss variable
     * @note 2: loss variable
     * @note 3: target variable
     */
    std::shared_ptr<Variable> getVariable(std::uint32_t index) override;
};

EnsembleModule::EnsembleModule(std::vector<std::shared_ptr<Variable>> inputVariables, const Loss & lossModule)
{
    mOutputVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Average>(), inputVariables, {})));

    mLossModule = std::make_shared<Module>(lossModule);

    // connections within the module
    mOutputVariable->getConsumers().push_back(mLossModule->getVariable(0)); // surrogate loss
    mOutputVariable->getConsumers().push_back(mLossModule->getVariable(1)); // loss
    mLossModule->__init__({mOutputVariable}, {});

    // other connections
    for(auto inputVariable : inputVariables)
    {
        inputVariable->getConsumers().push_back(mOutputVariable);
    }
}

// void EnsembleModule::remove()
// {
//     // delete other connections
//     for(auto input : mOutputVariable->getInputs())
//     {
//         input->getConsumers().erase(std::find(input->getConsumers().begin(), input->getConsumers().end(), mOutputVariable));
//     }

//     // delete the variables
//     GRAPH->removeVariable(mOutputVariable);
// }

std::shared_ptr<Variable> EnsembleModule::getVariable(std::uint32_t index)
{
    switch (index)
    {
    case 0:
        return mOutputVariable;
        break;
    case 1:
        return mLossModule->getVariable(0);
        break;
    case 2:
        return mLossModule->getVariable(1);
        break;
    case 3:
        return mLossModule->getVariable(2);
        break;
    default:
        throw std::invalid_argument("EnsembleModule::getVariable: index out of range");
        break;
    }
}

#endif // ENSEMBLE_MODULE_HPP
