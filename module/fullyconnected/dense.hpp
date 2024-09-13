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
     * @param name the name of the module
     * @param dropout the dropout rate of the layer
     */
    Dense(const std::shared_ptr<Operation> &activationFunction, std::uint32_t units, const std::string& name = "", const double& dropout = 1.0);

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

    void addInput(const std::shared_ptr<Variable> &input, const std::uint32_t &inputSize) override;
    void addOutput(const std::shared_ptr<Variable> &output) override;
};

inline Dense::Dense(const HiddenVariant &activationFunction, const std::uint32_t units, const std::string& name, const double& dropout) :
Dense(std::visit([]<typename T0>(T0&& arg) {
    // Assuming all types in the variant can be dynamically cast to Operation*
    return std::shared_ptr<Operation>(std::make_shared<std::decay_t<T0>>(arg));}, activationFunction), units, name, dropout)
{

}


inline Dense::Dense(const std::shared_ptr<Operation> &activationFunction, const std::uint32_t units, const std::string& name, const double& dropout) : FullyConnected(units, name)
{
    mpActivationVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(activationFunction, {mpMatmulVariable}, {})));
    
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

inline void Dense::addInput(const std::shared_ptr<Variable> &input, const std::uint32_t &inputSize)
{
    mpPaddingVariable->getInputs().push_back(input);

    // Initialize default norm if not already set
    if (!mpNorm && mpsDefaultNorm) {
        mpNorm = std::visit([]<typename T0>(T0&& arg) {
            return std::make_shared<std::decay_t<T0>>(std::forward<T0>(arg));
        }, *mpsDefaultNorm);
    }

    if (mpNorm != nullptr) // adding norm to activation function
    {
        mpNormVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(mpNorm, {mpWeightMatrixVariable}, {})));
        mpWeightMatrixVariable->getConsumers().push_back(mpNormVariable);
    }
}

inline void Dense::addOutput(const std::shared_ptr<Variable> &output)
{
    mpDropoutVariable->getConsumers().push_back(output);
}


#endif // DENSE_HPP