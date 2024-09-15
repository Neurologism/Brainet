#ifndef ENSEMBLE_MODULE_HPP
#define ENSEMBLE_MODULE_HPP

#include "module.hpp"
#include "loss.hpp"
#include "../operation/processing/average.hpp"


///////////////////////////////////////
///Not working
//////////////////////////////////////

/**
 * @brief the ensemble module is used to average the output of multiple Variables and apply a loss function to the output.
 */
class EnsembleModule final : Module
{
    std::shared_ptr<Variable> mAverageVariable; 
    std::shared_ptr<Module> mLossModule; 

public:
    /**
     * @brief constructor for the ensemble module
     * @param inputVariables the variables to average
     * @param name the name of the module
     * @param lossModule the loss module to apply to the output
     */
    EnsembleModule(const std::vector<std::shared_ptr<Variable>>& inputVariables, const std::string &name, const Loss & lossModule);

    
};

inline EnsembleModule::EnsembleModule(const std::vector<std::shared_ptr<Variable>>& inputVariables, const std::string &name, const Loss & lossModule) : Module(name)
{
    mAverageVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Average>(), inputVariables, {})));

    mLossModule = std::make_shared<Loss>(lossModule);

    // connections within the module
    mAverageVariable->getConsumers().push_back(mLossModule->getVariable(0)); // surrogate loss
    mAverageVariable->getConsumers().push_back(mLossModule->getVariable(1)); // loss
    mLossModule->addInput(mAverageVariable, 0);

    // other connections
    for(const auto& inputVariable : inputVariables)
    {
        inputVariable->getConsumers().push_back(mAverageVariable);
    }
}

inline std::shared_ptr<Variable> EnsembleModule::getVariable(std::uint32_t index)
{
    switch (index)
    {
    case 0:
        return mAverageVariable;
    case 1:
        return mLossModule->getVariable(0);
    case 2:
        return mLossModule->getVariable(1);
    case 3:
        return mLossModule->getVariable(2);
    default:
        throw std::invalid_argument("EnsembleModule::getVariable: index out of range");
    }
}

#endif // ENSEMBLE_MODULE_HPP
