#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include "fullyconnected.hpp"	
#include "../loss.hpp"

class Output : public FullyConnected
{
    std::shared_ptr<Loss> mpLoss;

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
    Output(OutputVariant activationFunction, std::uint32_t units, ParameterNormPenaltyVariant norm);

    /**
     * @brief add a output layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param units the number of neurons in the layer.
     * @param lossFunction the operation representing the loss function.
     */
    Output(OutputVariant activationFunction, std::uint32_t units, LossFunctionVariant lossFunction );

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

Output::Output(OutputVariant activationFunction, std::uint32_t units, ParameterNormPenaltyVariant norm) : FullyConnected(units)
{
    mpNorm = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to Operation*
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, norm);
    Output(activationFunction, units);
}

Output::Output(OutputVariant activationFunction, std::uint32_t units, LossFunctionVariant lossFunction ) : Output(activationFunction, units)
{
    mpLoss = std::make_shared<Loss>(Loss(lossFunction));

    // connect the loss function to the output layer

    mpLoss->__init__({mpActivationVariable}, {});
    mpActivationVariable->getConsumers().push_back(mpLoss->getVariable(0)); // surrogate loss
    mpActivationVariable->getConsumers().push_back(mpLoss->getVariable(1)); // loss

    std::shared_ptr<Operation> activation_function = mpActivationVariable->getOperation();
    std::shared_ptr<Operation> loss_function = mpLoss->getVariable(0)->getOperation();
    if ( dynamic_cast<Softmax*>(activation_function.get()) != nullptr && dynamic_cast<CrossEntropy*>(mpLoss->getVariable(0)->getOperation().get()) != nullptr )
    {
        ((Softmax*)activation_function.get())->useWithLog();
        ((CrossEntropy*)loss_function.get())->useWithExp();
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
            if (mpLoss == nullptr)
            {
                throw std::invalid_argument("Output::getVariable: loss variable not initialized");
            }
            return mpLoss->getVariable(0);
            break;
        case 5:
            if (mpLoss == nullptr)
            {
                throw std::invalid_argument("Output::getVariable: target variable not initialized");
            }
            return mpLoss->getVariable(1);
            break;
        case 6:
            if (mpLoss == nullptr)
            {
                throw std::invalid_argument("Output::getVariable: loss variable not initialized");
            }
            return mpLoss->getVariable(2);
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