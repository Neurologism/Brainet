//
// Created by servant-of-scietia on 20.09.24.
//
#include "module/dense.hpp"

std::shared_ptr<ParameterNormPenaltyVariant> Dense::mpsDefaultNorm = nullptr;

Dense::Dense(const ActivationVariant &activationFunction, const std::uint32_t units, const std::string& name, const double& dropout) :
Dense(std::visit([]<typename T0>(T0&& arg) {
    // Assuming all types in the variant can be dynamically cast to Operation*
    return std::shared_ptr<Operation>(std::make_shared<std::decay_t<T0>>(arg));}, activationFunction), units, name, dropout)
{

}


Dense::Dense(const std::shared_ptr<Operation> &activationFunction, const std::uint32_t size, const std::string& name, const double& dropout) : Layer(name)
{
    mSize = size; // set the number of neurons in the layer

    // create the variables
    mpDropoutVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Dropout>(Dropout(dropout)))));
    mpPaddingVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Padding>(Padding(0,1,1)), {mpDropoutVariable}))); // pad for weights

    mpWeightMatrixVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<WeightMatrixInitializer>(WeightMatrixInitializer(size, std::make_shared<NormalizedInitialization>(), std::dynamic_pointer_cast<ReLU>(activationFunction) ? 0.1 : 0)), {mpPaddingVariable})));
    mpMatmulVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Matmul>(Matmul()), {mpPaddingVariable,mpWeightMatrixVariable})));
    mpActivationVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(activationFunction, {mpMatmulVariable})));

    // connections within the module
    mpDropoutVariable->getConsumers().push_back(mpPaddingVariable);
    mpPaddingVariable->getConsumers().push_back(mpMatmulVariable);
    mpPaddingVariable->getConsumers().push_back(mpWeightMatrixVariable);
    mpWeightMatrixVariable->getConsumers().push_back(mpMatmulVariable);
    mpMatmulVariable->getConsumers().push_back(mpActivationVariable);

    // Initialize default norm if not already set
    // if (!mpNorm && mpsDefaultNorm != nullptr) {
    //     mpNorm = std::visit([]<typename T0>(T0&& arg) {
    //         return std::make_shared<std::decay_t<T0>>(std::forward<T0>(arg));
    //     }, *mpsDefaultNorm);
    // }

    if (mpNorm != nullptr) // adding norm to activation function
    {
        mpNormVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(mpNorm, {mpWeightMatrixVariable}, {})));
        mpWeightMatrixVariable->getConsumers().push_back(mpNormVariable);
    }
}

Dense::Dense(const ActivationVariant& activationFunction, const std::uint32_t size, ParameterNormPenaltyVariant norm, const std::string& name, const double dropout) : Dense(activationFunction, size, name, dropout)
{
    mpNorm = std::visit([]<typename T0>(T0&& arg) {
        // Assuming all types in the variant can be dynamically cast to Operation*
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<T0>>(arg));}, norm);

}

std::vector<std::shared_ptr<Variable>> Dense::getInputs()
{
    return {mpDropoutVariable};
}

std::vector<std::shared_ptr<Variable>> Dense::getOutputs()
{
    return {mpActivationVariable};
}

std::vector<std::shared_ptr<Variable>> Dense::getLearnableVariables()
{
    if (mpNormVariable != nullptr)
    {
        return {mpWeightMatrixVariable, mpNormVariable};
    }
    return {mpWeightMatrixVariable};
}

std::vector<std::shared_ptr<Variable>> Dense::getGradientVariables()
{
    if (mpNormVariable != nullptr)
    {
        return {mpNormVariable};
    }
    return {};
}

void Dense::setDefaultNorm(ParameterNormPenaltyVariant const & norm)
{
    mpsDefaultNorm = std::make_shared<ParameterNormPenaltyVariant>(norm);
}



