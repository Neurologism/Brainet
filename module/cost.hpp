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
    std::shared_ptr<Variable> mOutputVariable; // actual cost funtion 
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

    ~Cost() = default;

    /**
     * @brief used to mark variables as input for the module.
     */
    void addInput(std::shared_ptr<Variable> input, std::uint32_t units) override
    {
        mOutputVariable->getInputs().push_back(input);
    }
    /**
     * @brief used to mark variables as output for the module.
     */
    void addOutput(std::shared_ptr<Variable> output) override
    {
        mOutputVariable->getConsumers().push_back(output);
    }
    /**
     * @brief used to get the input variables of the module specified by the index.
     */
    std::shared_ptr<Variable> input(std::uint32_t index) override
    {
        return mOutputVariable;
    }
    /**
     * @brief used to get the output variables of the module specified by the index.
     */
    std::shared_ptr<Variable> output(std::uint32_t index) override
    {
        return mOutputVariable;
    }
    /**
     * @brief used to get the target variable of the module.
     */
    std::shared_ptr<Variable> target()
    {
        return mTargetVariable;
    }
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
    mOutputVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::visit([](auto&& arg) {
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, CostVariant{costFunction}), {mOneHotVariable}, {})));
    
    // connections within the module
    mTargetVariable->getConsumers().push_back(mOneHotVariable);
    mOneHotVariable->getConsumers().push_back(mOutputVariable);
}

Cost::Cost(CostVariant costFunction)
{
    // error checks
    if(GRAPH == nullptr)
    {
        throw std::runtime_error("graph is not set");
    }


    // add variables to the graph
    mTargetVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {})));

    mOneHotVariable = nullptr;
        
    // add variables to the graph
    mTargetVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {})));

    mOutputVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::visit([](auto&& arg) {
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, CostVariant{costFunction}), {mTargetVariable}, {})));
    
    // connections within the module
    mTargetVariable->getConsumers().push_back(mOutputVariable);
}
#endif // COST_HPP