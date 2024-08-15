#ifndef COST_HPP
#define COST_HPP

#include "../dependencies.hpp"
#include "module.hpp"
#include "../operation/cost_function/cost_function.hpp"
#include "../operation/processing/one_hot.hpp"

/**
 * @brief the cost module is intended for calculating the cost of various models. Currently only cost functions dependy on the output of the network and the y truth are supported.
 
 */
class Cost : public Module
{
    std::shared_ptr<Variable> mTargetVariable; // storing y truth
    std::shared_ptr<Variable> mCostVariable; // actual cost funtion 
    std::shared_ptr<Variable> mOneHotVariable; // one hot encoding of the y truth

public:
    /**
     * @brief add a cost function to the graph
     * @param costFunction the operation representing the cost function.
     */
    Cost(CostVariant costFunction);

    /**
     * @brief add a cost function to the graph
     * @param costFunction the operation representing the cost function.
     * @param encodingSize the size of the one hot encoding. Default is 0. One hot encoding assumes that the y truth is initially a single value giving the index.
     * @param labelSmoothing the value to smooth the labels. Default is 0.
     */
    Cost(CostVariant costFunction, std::uint32_t encodingSize, double labelSmoothing = 0);

    ~Cost();

    void __init__( std::vector<std::shared_ptr<Variable>> initialInpus, std::vector<std::shared_ptr<Variable>> initialOutputs ) override;

    /**
     * @brief function to get access to specific variables of the module.
     * @param index the index of the variable
     * @return the variable specified by the index
     * @note 0: cost variable
     * @note 1: cost variable
     * @note 2: target variable
     */
    std::shared_ptr<Variable> getVariable(std::uint32_t index) override;
    
};

Cost::Cost(CostVariant costFunction, std::uint32_t encodingSize, double labelSmoothing)
{
    if (labelSmoothing < 0)
    {
        throw std::invalid_argument("Cost::Cost: labelSmoothing must be at least 0");
    }
    if (labelSmoothing >= 1)
    {
        throw std::invalid_argument("Cost::Cost: labelSmoothing must be less than 1");
    }
    
    // error checks
    if(GRAPH == nullptr)
    {
        throw std::runtime_error("graph is not set");
    }

    // add variables to the graph
    mTargetVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {})));

    
    // variable performing one hot encoding; no backward pass is supported
    mOneHotVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<OneHot>(OneHot(encodingSize, 1-labelSmoothing, labelSmoothing/(encodingSize-1))), {mTargetVariable}, {}))); 

    // conversion of the CostVariant to an operation pointer
    mCostVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::visit([](auto&& arg) {
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, CostVariant{costFunction}), {mOneHotVariable}, {})));
    
    // connections within the module
    mTargetVariable->getConsumers().push_back(mOneHotVariable);
    mOneHotVariable->getConsumers().push_back(mCostVariable);
}

Cost::Cost(CostVariant costFunction)
{
    // add variables to the graph
    mTargetVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {})));

    mOneHotVariable = nullptr;
        
    // add variables to the graph
    mTargetVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {})));

    mCostVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::visit([](auto&& arg) {
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, CostVariant{costFunction}), {mTargetVariable}, {})));
    
    // connections within the module
    mTargetVariable->getConsumers().push_back(mCostVariable);
}

Cost::~Cost()
{
    // delete other connections
    for(auto input : mCostVariable->getInputs())
    {
        input->getConsumers().erase(std::find(input->getConsumers().begin(), input->getConsumers().end(), mCostVariable));
    }

    // delete the variables
    GRAPH->removeVariable(mTargetVariable);
    if (mOneHotVariable != nullptr)
    {
        GRAPH->removeVariable(mOneHotVariable);
    }
    GRAPH->removeVariable(mCostVariable);
}


void Cost::__init__( std::vector<std::shared_ptr<Variable>> initialInpus, std::vector<std::shared_ptr<Variable>> initialOutputs )
{
    if (initialInpus.size() != 1)
    {
        throw std::invalid_argument("Cost::__init__: the number of input variables must be 1");
    }
    if (initialOutputs.size() != 0)
    {
        throw std::invalid_argument("Cost::__init__: the number of output variables must be 0");
    }

    mCostVariable->getInputs().push_back(initialInpus[0]);
}

std::shared_ptr<Variable> Cost::getVariable(std::uint32_t index)
{
    switch (index)
    {
    case 0:
        return mCostVariable;
        break;
    case 1:
        return mCostVariable;
        break;
    case 2:
        return mTargetVariable;
        break;
    default:
        throw std::invalid_argument("Cost::getVariable: index out of range");
        break;
    }
}



#endif // COST_HPP