#ifndef DENSE_HPP
#define DENSE_HPP

#include "./fullyconnected.hpp"

/**
 * @brief the dense module is intended for creating a dense (fully connected) layer in the graph. It owns 1 input and 1 output variable.
 */
class Dense : public FullyConnected
{
    std::shared_ptr<Variable> mpDropoutVariable;

public:
    /**
     * @brief add a dense layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param units the number of neurons in the layer.
     */
    Dense(HiddenVariant activationFunction, std::uint32_t units, double dropout = 1.0);

    /**
     * @brief add a dense layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param units the number of neurons in the layer.
     * @param norm the norm to use for regularization.
     */
    Dense(HiddenVariant activationFunction, std::uint32_t units, ParameterNormPenaltyVariant norm, double dropout = 1.0);

    ~Dense() = default;

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
     * @param initialInpus the initial input variables
     * @param initialOutputs the initial output variables
     */
    void __init__(std::vector<std::shared_ptr<Variable>> initialInpus, std::vector<std::shared_ptr<Variable>> initialOutputs) override;
};

Dense::Dense(HiddenVariant activationFunction, std::uint32_t units, double dropout) : FullyConnected(units)
{
    // turning the variant into a shared pointer to the operation class
    // Use std::visit to handle the variant
    std::shared_ptr<Operation> activationFunctionPtr = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to Operation*
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, HiddenVariant{activationFunction});

    mpActivationVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(activationFunctionPtr, {mpMatmulVariable}, {}))); 
    
    mpDropoutVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Dropout>(Dropout(dropout)), {mpActivationVariable}, {})));

    // connections within the module
    mpMatmulVariable->getConsumers().push_back(mpActivationVariable); 
    mpActivationVariable->getConsumers().push_back(mpDropoutVariable);
   
}

Dense::Dense(HiddenVariant activationFunction, std::uint32_t units, ParameterNormPenaltyVariant norm, double dropout) : Dense(activationFunction, units, dropout)
{
    mpNorm = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to Operation*
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, norm);
    
}

std::shared_ptr<Variable> Dense::getVariable(std::uint32_t index)
{
    switch (index)
    {
    case 0:
        return mpPaddingVariable;
        break;
    case 1:
        return mpDropoutVariable;
        break;
    case 2:
        return mpWeightMatrixVariable;
        break;
    case 3:
        if (mpNormVariable == nullptr)
        {
            throw std::invalid_argument("Dense::getVariable: norm variable not initialized");
        }
        return mpNormVariable;
        break;
    default:
        throw std::invalid_argument("Dense::getVariable: index out of range");
        break;
    }
}

void Dense::__init__(std::vector<std::shared_ptr<Variable>> initialInpus, std::vector<std::shared_ptr<Variable>> initialOutputs)
{
    if (initialInpus.size() != 1)
    {
        throw std::invalid_argument("Dense::__init__: the number of input variables must be 1");
    }
    if (initialOutputs.size() != 1)
    {
        throw std::invalid_argument("Dense::__init__: the number of output variables must be 1");
    }

    mpPaddingVariable->getInputs().push_back(initialInpus[0]);
        
    // init default norm
    if(mpNorm == nullptr && mpsDefaultNorm != nullptr)
    {
        mpNorm = std::visit([](auto&& arg) {
            // Assuming all types in the variant can be dynamically casted to Operation*
            return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, *mpsDefaultNorm);
    }
    
    if (mpNorm != nullptr) // adding norm to activation function
    {
        mpNormVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(mpNorm, {mpWeightMatrixVariable}, {})));  
        mpWeightMatrixVariable->getConsumers().push_back(mpNormVariable);
    }


    mpDropoutVariable->getConsumers().push_back(initialOutputs[0]);
}

#endif // DENSE_HPP