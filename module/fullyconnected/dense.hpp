#ifndef DENSE_HPP
#define DENSE_HPP

#include "./fullyconnected.hpp"

/**
 * @brief the dense module is intended for creating a dense (fully connected) layer in the graph. It owns 1 input and 1 output variable.
 */
class Dense : public FullyConnected
{


public:
    /**
     * @brief add a dense layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param units the number of neurons in the layer.
     */
    Dense(HiddenVariant activationFunction, std::uint32_t units);

    /**
     * @brief add a dense layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param units the number of neurons in the layer.
     * @param norm the norm to use for regularization.
     */
    Dense(HiddenVariant activationFunction, std::uint32_t units, NormVariant norm);

    ~Dense() = default;

    /**
     * @brief function to get access to specific variables of the module.
     * @param index the index of the variable
     * @return the variable specified by the index
     * @note 0: padding variable
     * @note 1: activation variable
     * @note 2: weight matrix variable
     * @note 3: norm variable
     */
    std::shared_ptr<Variable> getVariable(std::uint32_t index) override;
};

Dense::Dense(HiddenVariant activationFunction, std::uint32_t units) : FullyConnected(units)
{
    // turning the variant into a shared pointer to the operation class
    // Use std::visit to handle the variant
    std::shared_ptr<Operation> activationFunctionPtr = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to Operation*
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, HiddenVariant{activationFunction});

    mpActivationVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(activationFunctionPtr, {mpMatmulVariable}, {})));

    // connections within the module
    mpMatmulVariable->getConsumers().push_back(mpActivationVariable);    
}

Dense::Dense(HiddenVariant activationFunction, std::uint32_t units, NormVariant norm) : FullyConnected(units)
{
    mpNorm = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to Operation*
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, norm);
    Dense(activationFunction, units);
}

std::shared_ptr<Variable> Dense::getVariable(std::uint32_t index)
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
    default:
        throw std::invalid_argument("FullyConnected::getVariable: index out of range");
        break;
    }
}

#endif // DENSE_HPP