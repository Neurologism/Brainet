#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include "./fullyconnected.hpp"

class Output : public FullyConnected
{
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

    ~Output() = default;
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


#endif // OUTPUT_HPP