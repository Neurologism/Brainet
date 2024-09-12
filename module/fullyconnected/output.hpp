#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include <graph.hpp>

#include "fullyconnected.hpp"
#include "../loss.hpp"

class Output : public FullyConnected
{
    std::shared_ptr<Loss> mpLoss;

public:
    /**
     * @brief add an output layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param units the number of neurons in the layer.
     * @param name the name of the module
     */
    Output(const OutputVariant &activationFunction, std::uint32_t units, const std::string& name = "");


    /**
     * @brief add an output layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param units the number of neurons in the layer.
     * @param norm the norm to use for regularization.
     */
    // Output(const OutputVariant& activationFunction, std::uint32_t units, ParameterNormPenaltyVariant norm);

    /**
     * @brief add an output layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param units the number of neurons in the layer.
     * @param lossFunction the operation representing the loss function.
     */
    // Output(const OutputVariant& activationFunction, std::uint32_t units, const LossFunctionVariant& lossFunction );

    ~Output() = default;

    /**
     * @brief function to get access to specific variables of the module.
     * @param index the index of the variable
     * @return the variable specified by the index
     * @note 0: padding variable
     * @note 1: activation variable
     * @note 2: weight matrix variable
     * @note 3: norm variable
     * @note 4: surrogate loss variable
     * @note 5: loss variable
     * @note 6: target variable
     */
    std::shared_ptr<Variable> getVariable(std::uint32_t index);

    /**
     * @brief used to initialize the module with the input and output variables.
     */
    void __init__(const std::vector<std::shared_ptr<Variable>> &initialInputs, const std::vector<std::shared_ptr<Variable>>& initialOutputs) const;
};

inline Output::Output(const OutputVariant &activationFunction, const std::uint32_t units, const std::string& name) : FullyConnected(units, name)
{
    // turning the variant into a shared pointer to the operation class
    // Use std::visit to handle the variant
    std::shared_ptr<Operation> activationFunctionPtr = std::visit([]<typename T0>(T0&& arg) {
        // Assuming all types in the variant can be dynamically cast to Operation*
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<T0>>(arg));}, OutputVariant{activationFunction});

    mpActivationVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(activationFunctionPtr, {mpMatmulVariable}, {})));

    // connections within the module
    mpMatmulVariable->getConsumers().push_back(mpActivationVariable);    
}

// inline Output::Output(const OutputVariant& activationFunction, const std::uint32_t units, ParameterNormPenaltyVariant norm) : FullyConnected(units)
// {
//     mpNorm = std::visit([]<typename T0>(T0&& arg) {
//         // Assuming all types in the variant can be dynamically cast to Operation*
//         return std::shared_ptr<Operation>(std::make_shared<std::decay_t<T0>>(arg));}, norm);
//     Output(activationFunction, units);
// }

// inline Output::Output(const OutputVariant& activationFunction, const std::uint32_t units, const LossFunctionVariant& lossFunction ) : Output(activationFunction, units)
// {
//     mpLoss = std::make_shared<Loss>(Loss(lossFunction));
//
//     // connect the loss function to the output layer
//
//     mpLoss->__init__({mpActivationVariable}, {});
//     mpActivationVariable->getConsumers().push_back(mpLoss->getVariable(0)); // surrogate loss
//     mpActivationVariable->getConsumers().push_back(mpLoss->getVariable(1)); // loss
//
//     std::shared_ptr<Operation> activation_function = mpActivationVariable->getOperation();
//     std::shared_ptr<Operation> loss_function = mpLoss->getVariable(0)->getOperation();
//     if ( dynamic_cast<Softmax*>(activation_function.get()) != nullptr && dynamic_cast<CrossEntropy*>(mpLoss->getVariable(0)->getOperation().get()) != nullptr )
//     {
//         dynamic_cast<Softmax *>(activation_function.get())->useWithLog();
//         dynamic_cast<CrossEntropy *>(loss_function.get())->useWithExp();
//     }
//
// }

inline std::shared_ptr<Variable> Output::getVariable(const std::uint32_t index)
{
    switch (index)
    {
        case 0:
            return mpPaddingVariable;
        case 1:
            return mpActivationVariable;
        case 2:
            return mpWeightMatrixVariable;
        case 3: 
            if (mpNormVariable == nullptr)
            {
                throw std::invalid_argument("FullyConnected::getVariable: norm variable not initialized");
            }
            return mpNormVariable;
        case 4:
            if (mpLoss == nullptr)
            {
                throw std::invalid_argument("Output::getVariable: loss variable not initialized");
            }
            return mpLoss->getVariable(0);
        case 5:
            if (mpLoss == nullptr)
            {
                throw std::invalid_argument("Output::getVariable: target variable not initialized");
            }
            return mpLoss->getVariable(1);
        case 6:
            if (mpLoss == nullptr)
            {
                throw std::invalid_argument("Output::getVariable: loss variable not initialized");
            }
            return mpLoss->getVariable(2);
        default:
            throw std::invalid_argument("Output::getVariable: invalid index");
    }
}

inline void Output::__init__(const std::vector<std::shared_ptr<Variable>> &initialInputs, const std::vector<std::shared_ptr<Variable>>& initialOutputs) const
{
    if (initialInputs.size() != 1)
    {
        throw std::invalid_argument("Output::__init__: the number of input variables must be 1");
    }
    if (!initialOutputs.empty())
    {
        throw std::invalid_argument("Output::__init__: the number of output variables must be 0");
    }

    mpPaddingVariable->getInputs().push_back(initialInputs[0]);
}


#endif // OUTPUT_HPP