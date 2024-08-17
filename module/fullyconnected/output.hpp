#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include "./fullyconnected.hpp"	
#include "../cost.hpp"

class Output : public FullyConnected
{
    std::shared_ptr<Cost> mpCost;

public:
    /**
     * @brief add a output layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param units the number of neurons in the layer.
     */
    Output(OutputVariant activationFunction, std::uint32_t units);

    /**
     * @brief add a output layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param units the number of neurons in the layer.
     * @param norm the norm to use for regularization.
     */
    Output(OutputVariant activationFunction, std::uint32_t units, NormVariant norm);

    /**
     * @brief add a output layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param units the number of neurons in the layer.
     * @param costFunction the operation representing the cost function.
     */
    Output(OutputVariant activationFunction, std::uint32_t units, CostVariant costFunction );

    ~Output() = default;

    /**
     * @brief function to get access to specific variables of the module.
     * @param index the index of the variable
     * @return the variable specified by the index
     * @note 0: padding variable
     * @note 1: activation variable
     * @note 2: weight matrix variable
     * @note 3: norm variable
     * @note 4: cost variable
     * @note 5: target variable
     */
    std::shared_ptr<Variable> getVariable(std::uint32_t index) override;

    /**
     * @brief used to initialize the module with the input and output variables.
     */
    void __init__(std::vector<std::shared_ptr<Variable>> initialInpus, std::vector<std::shared_ptr<Variable>> initialOutputs) override;
};

Output::Output(OutputVariant activationFunction, std::uint32_t units) : FullyConnected(units)
{
    // turning the variant into a shared pointer to the operation class
    // Use std::visit to handle the variant
    std::shared_ptr<Operation> activationFunctionPtr = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to Operation*
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, OutputVariant{activationFunction});

    mpActivationVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(activationFunctionPtr, {mpMatmulVariable}, {})));

    // connections within the module
    mpMatmulVariable->getConsumers().push_back(mpActivationVariable);    
}

Output::Output(OutputVariant activationFunction, std::uint32_t units, NormVariant norm) : FullyConnected(units)
{
    mpNorm = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to Operation*
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, norm);
    Output(activationFunction, units);
}

Output::Output(OutputVariant activationFunction, std::uint32_t units, CostVariant costFunction ) : Output(activationFunction, units)
{
    mpCost = std::make_shared<Cost>(costFunction);

    // connect the cost function to the output layer

    mpCost->__init__({mpActivationVariable}, {});
    mpActivationVariable->getConsumers().push_back(mpCost->getVariable(0));

    std::shared_ptr<Operation> activation_function = mpActivationVariable->getOperation();
    std::shared_ptr<Operation> cost_function = mpCost->getVariable(0)->getOperation();
    if ( dynamic_cast<Softmax*>(activation_function.get()) != nullptr && dynamic_cast<CrossEntropy*>(cost_function.get()) != nullptr) // numerical stability
    {
        ((Softmax*)activation_function.get())->useWithLog();
        ((CrossEntropy*)cost_function.get())->useWithExp();
    }

}

std::shared_ptr<Variable> Output::getVariable(std::uint32_t index)
{
    switch (index)
    {
        case 0:
            return mpPaddingVariable;
            break;
        case 1:
            return mpActivationVariable;
            break;
        case 2:
            return mpWeightMatrixVariable;
            break;
        case 3: 
            if (mpNormVariable == nullptr)
            {
                throw std::invalid_argument("FullyConnected::getVariable: norm variable not initialized");
            }
            return mpNormVariable;
            break;
        case 4:
            if (mpCost == nullptr)
            {
                throw std::invalid_argument("Output::getVariable: cost variable not initialized");
            }
            return mpCost->getVariable(0);
            break;
        case 5:
            if (mpCost == nullptr)
            {
                throw std::invalid_argument("Output::getVariable: target variable not initialized");
            }
            return mpCost->getVariable(2);
            break;
        default:
            throw std::invalid_argument("Output::getVariable: invalid index");
            break;
    }
}

void Output::__init__(std::vector<std::shared_ptr<Variable>> initialInpus, std::vector<std::shared_ptr<Variable>> initialOutputs)
{
    if (initialInpus.size() != 1)
    {
        throw std::invalid_argument("Output::__init__: the number of input variables must be 1");
    }
    if (initialOutputs.size() != 0)
    {
        throw std::invalid_argument("Output::__init__: the number of output variables must be 0");
    }

    mpPaddingVariable->getInputs().push_back(initialInpus[0]);
}


#endif // OUTPUT_HPP