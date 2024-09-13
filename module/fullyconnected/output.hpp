#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include <graph.hpp>

#include "fullyconnected.hpp"
#include "../loss.hpp"

class Output : public FullyConnected
{
    std::shared_ptr<Loss> mpLoss;

public:



    // /**
    //  * @brief add an output layer to the graph
    //  * @param activationFunction the operation representing the activation function.
    //  * @param units the number of neurons in the layer.
    //  * @param name
    //  * @param norm the norm to use for regularization.
    //  */
    // // Output(const OutputVariant& activationFunction, std::uint32_t units, ParameterNormPenaltyVariant norm);

    /**
     * @brief add an output layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param units the number of neurons in the layer.
     * @param lossFunction the operation representing the loss function.
     * @param name the name of the module
     */
    Output(const OutputVariant &activationFunction, std::uint32_t units, const LossFunctionVariant &lossFunction,
               const std::string &name = "");


    Output(const std::shared_ptr<Operation> &activationFunction, std::uint32_t units, const LossFunctionVariant &lossFunction,
                const std::string &name = "");
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

    void addInput(const std::shared_ptr<Variable> &input, const std::uint32_t &inputSize) override;
    void addOutput(const std::shared_ptr<Variable> &output) override;
};

inline Output::Output(const OutputVariant& activationFunction, const std::uint32_t units, const LossFunctionVariant& lossFunction, const std::string& name ) :
    Output(std::visit([]<typename T0>(T0&& arg) {
        // Assuming all types in the variant can be dynamically cast to Operation*
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<T0>>(arg));}, activationFunction), units, lossFunction, name)
{

}

inline Output::Output(const std::shared_ptr<Operation> &activationFunction, std::uint32_t units, const LossFunctionVariant &lossFunction, const std::string &name) : FullyConnected(units, name)
{
    mpActivationVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(activationFunction, {mpMatmulVariable}, {})));

    // connections within the module
    mpMatmulVariable->getConsumers().push_back(mpActivationVariable);

    mpLoss = std::make_shared<Loss>(Loss(lossFunction));

    // connect the loss function to the output layer

    mpLoss->__init__({mpActivationVariable}, {});
    mpActivationVariable->getConsumers().push_back(mpLoss->getVariable(0)); // surrogate loss
    mpActivationVariable->getConsumers().push_back(mpLoss->getVariable(1)); // loss

    const std::shared_ptr<Operation> loss_function = mpLoss->getVariable(0)->getOperation();
    if ( dynamic_cast<Softmax*>(activationFunction.get()) != nullptr && dynamic_cast<CrossEntropy*>(mpLoss->getVariable(0)->getOperation().get()) != nullptr )
    {
        dynamic_cast<Softmax *>(activationFunction.get())->useWithLog();
        dynamic_cast<CrossEntropy *>(loss_function.get())->useWithExp();
    }
}


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

inline void Output::addInput(const std::shared_ptr<Variable> &input, const std::uint32_t & inputSize)
{
    mpPaddingVariable->getInputs().push_back(input);
    createWeightMatrix(inputSize);
}

inline void Output::addOutput(const std::shared_ptr<Variable>& output)
{
    throw std::runtime_error("Output::addOutput: output module cannot have outputs");
}

#endif // OUTPUT_HPP