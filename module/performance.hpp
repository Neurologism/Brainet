#ifndef PERFORMANCE_MODULE_HPP
#define PERFORMANCE_MODULE_HPP

#include "module.hpp"
#include "../operation/performance/performance_function.hpp"

/**
 * @brief the performance module is used to evaluate the performance of the model.
 */
class Performance : public Module
{
    std::shared_ptr<Variable> mTargetVariable;
    std::shared_ptr<Variable> mPerformanceVariable;

public:
    /**
     * @brief constructor for the performance module
     * @param performanceFunction the performance function to apply to the output
     */
    Performance(PerformanceFunctionVariant performanceFunction);

    ~Performance();
    
    /**
     * @brief function to initialize the module
     * @param initialInpus the initial input variables
     * @param initialOutputs the initial output variables
     * @note initialInpus[0]: prediction variable
     * @note initialInpus[1]: target variable (optional)
     */
    void __init__( std::vector<std::shared_ptr<Variable>> initialInpus, std::vector<std::shared_ptr<Variable>> initialOutputs ) override;

    /**
     * @brief function to get access to specific variables of the module.
     * @param index the index of the variable
     * @return the variable specified by the index
     * @note 0: performance variable
     * @note 1: performance variable
     * @note 2: target variable
     */
    std::shared_ptr<Variable> getVariable(std::uint32_t index) override;

};

Performance::Performance(PerformanceFunctionVariant performanceFunction)
{
    std::shared_ptr<Operation> operation = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to Operation*
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, performanceFunction);
    
    mPerformanceVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(operation, {}, {})));
}

Performance::~Performance()
{
    // to be implemented
}

void Performance::__init__( std::vector<std::shared_ptr<Variable>> initialInpus, std::vector<std::shared_ptr<Variable>> initialOutputs )
{
    if(initialOutputs.size() != 0)
    {
        throw std::runtime_error("Performance::__init__: initialOutputs must be empty");
    }
    if(initialInpus.size() == 2) 
    {
        mTargetVariable = initialInpus[1];
    }
    else
    {
        mTargetVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {})));
    }

    mPerformanceVariable->getInputs().push_back(initialInpus[0]);
    mPerformanceVariable->getInputs().push_back(mTargetVariable);

    initialInpus[0]->getConsumers().push_back(mPerformanceVariable);
    mTargetVariable->getConsumers().push_back(mPerformanceVariable);
}

std::shared_ptr<Variable> Performance::getVariable(std::uint32_t index)
{
    switch (index)
    {
    case 0:
        return mPerformanceVariable;
    case 1:
        return mPerformanceVariable;
    case 2:
        return mTargetVariable;
    default:
        throw std::invalid_argument("Performance::getVariable: index out of range");
    }
}

#endif // PERFORMANCE_MODULE_HPP