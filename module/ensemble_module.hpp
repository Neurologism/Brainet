#ifndef ENSEMBLE_MODULE_HPP
#define ENSEMBLE_MODULE_HPP

#include "module.hpp"
#include "loss.hpp"
#include "../operation/processing/average.hpp"


/**
 * @brief the ensemble module is used to average the output of multiple Variables and apply a loss function to the output.
 */
class EnsembleModule final : private Module
{
    std::shared_ptr<Variable> mOutputVariable; 
    std::shared_ptr<Module> mLossModule; 
    
    // not supported
    void addInput(const std::shared_ptr<Variable> &input, const std::uint32_t &inputSize) override {}
    void addOutput(const std::shared_ptr<Variable>& output) override {}

public:
    /**
     * @brief constructor for the ensemble module
     * @param inputVariables the variables to average
     * @param name the name of the module
     * @param lossModule the loss module to apply to the output
     */
    EnsembleModule(const std::vector<std::shared_ptr<Variable>>& inputVariables, const std::string &name, const Loss & lossModule);

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

inline EnsembleModule::EnsembleModule(const std::vector<std::shared_ptr<Variable>>& inputVariables, const std::string &name, const Loss & lossModule) : Module(name)
{
    mOutputVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Average>(), inputVariables, {})));

    mLossModule = std::make_shared<Loss>(lossModule);

    // connections within the module
    mOutputVariable->getConsumers().push_back(mLossModule->getVariable(0)); // surrogate loss
    mOutputVariable->getConsumers().push_back(mLossModule->getVariable(1)); // loss
    mLossModule->addInput(mOutputVariable, 0);

    // other connections
    for(const auto& inputVariable : inputVariables)
    {
        inputVariable->getConsumers().push_back(mOutputVariable);
    }
}

inline std::shared_ptr<Variable> EnsembleModule::getVariable(std::uint32_t index)
{
    switch (index)
    {
    case 0:
        return mOutputVariable;
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
