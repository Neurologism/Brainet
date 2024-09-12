#ifndef DENSE_HPP
#define DENSE_HPP

#include <operation/processing/dropout.hpp>

#include "fullyconnected.hpp"

/**
 * @brief The dense module is intended for creating a dense (fully connected) layer in the graph.
 * It owns one input and one output variable.
 */
class Dense final : public FullyConnected
{
    std::shared_ptr<Variable> mpDropoutVariable;

public:
    /**
     * @brief add a dense layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param units the number of neurons in the layer.
     * @param name the name of the module
     * @param dropout the dropout rate of the layer
     */
    Dense(const HiddenVariant &activationFunction, std::uint32_t units, const std::string& name = "", const double& dropout = 1.0);

    /**
     * @brief add a dense layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param units the number of neurons in the layer.
     * @param norm the norm to use for regularization.
     * @param name the name of the module
     * @param dropout the dropout rate of the layer
     */
    Dense(const HiddenVariant& activationFunction, std::uint32_t units, ParameterNormPenaltyVariant norm, const std::string& name = "", double dropout = 1.0);

    ~Dense() override = default;

    /**
     * @brief function to get access to specific variables of the module.
     * @param index the index of the variable
     * @return the variable specified by the index
     * @note 0: padding variable
     * @note 1: dropout variable
     * @note 2: weight matrix variable
     * @note 3: norm variable
     */
    std::shared_ptr<Variable> getVariable(std::uint32_t index) override;

    /**
     * @brief function to initialize the module
     * @param initialInputs the initial input variables
     * @param initialOutputs the initial output variables
     */
    void __init__(std::vector<std::shared_ptr<Variable>> initialInputs, std::vector<std::shared_ptr<Variable>> initialOutputs) override;
};

inline Dense::Dense(const HiddenVariant &activationFunction, const std::uint32_t units, const std::string& name, const double& dropout) : FullyConnected(units, name)
{
    // turning the variant into a shared pointer to the operation class
    // Use std::visit to handle the variant
    const std::shared_ptr<Operation> activationFunctionPtr = std::visit([]<typename T0>(T0&& arg) {
        // Assuming all types in the variant can be dynamically cast to Operation*
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<T0>>(arg));}, HiddenVariant{activationFunction});

    mpActivationVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(activationFunctionPtr, {mpMatmulVariable}, {}))); 
    
    mpDropoutVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Dropout>(Dropout(dropout)), {mpActivationVariable}, {})));

    // connections within the module
    mpMatmulVariable->getConsumers().push_back(mpActivationVariable); 
    mpActivationVariable->getConsumers().push_back(mpDropoutVariable);
   
}

inline Dense::Dense(const HiddenVariant& activationFunction, const std::uint32_t units, ParameterNormPenaltyVariant norm, const std::string& name, const double dropout) : Dense(activationFunction, units, name, dropout)
{
    mpNorm = std::visit([]<typename T0>(T0&& arg) {
        // Assuming all types in the variant can be dynamically cast to Operation*
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<T0>>(arg));}, norm);
    
}

inline std::shared_ptr<Variable> Dense::getVariable(const std::uint32_t index)
{
    switch (index)
    {
    case 0:
        return mpPaddingVariable;
    case 1:
        return mpDropoutVariable;
    case 2:
        return mpWeightMatrixVariable;
    case 3:
        if (mpNormVariable == nullptr)
        {
            throw std::invalid_argument("Dense::getVariable: norm variable not initialized");
        }
        return mpNormVariable;
    default:
        throw std::invalid_argument("Dense::getVariable: index out of range");
    }
}

void Dense::__init__(const std::vector<std::shared_ptr<Variable>> initialInputs, const std::vector<std::shared_ptr<Variable>> initialOutputs)
{
    if (initialInputs.size() != 1)
    {
        throw std::invalid_argument("Dense::__init__: the number of input variables must be 1");
    }
    if (initialOutputs.size() != 1)
    {
        throw std::invalid_argument("Dense::__init__: the number of output variables must be 1");
    }

    mpPaddingVariable->getInputs().push_back(initialInputs[0]);
        
    // init default norm
    if(mpNorm == nullptr && mpsDefaultNorm != nullptr)
    {
        mpNorm = std::visit([]<typename T0>(T0&& arg) {
            // Assuming all types in the variant can be dynamically cast to Operation*
            return std::shared_ptr<Operation>(std::make_shared<std::decay_t<T0>>(arg));}, *mpsDefaultNorm);
    }
    
    if (mpNorm != nullptr) // adding norm to activation function
    {
        mpNormVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(mpNorm, {mpWeightMatrixVariable}, {})));  
        mpWeightMatrixVariable->getConsumers().push_back(mpNormVariable);
    }


    mpDropoutVariable->getConsumers().push_back(initialOutputs[0]);
}

#endif // DENSE_HPP