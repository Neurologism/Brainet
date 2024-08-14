#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include "./module.hpp"

class Output : public Module
{
    // storing index of the variables in the graph
    std::shared_ptr<Variable> mpWeightMatrixVariable; // learnable parameters of the layer (weights + bias)
    std::shared_ptr<Variable> mpMatmulVariable; // multiplication of the input and the weights
    std::shared_ptr<Variable> mpActivationVariable; // activation function applied
    std::shared_ptr<Variable> mpPaddingVariable; // used to pad the input with 1s for the bias
    std::shared_ptr<Variable> mpNormVariable; // used to compute a norm of the weights

    static std::shared_ptr<NormVariant> mpsDefaultNorm; // default norm to use
    std::shared_ptr<Operation> mpNorm = nullptr; // norm to use for regularization



public:
    /**
     * @brief add a dense layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param units the number of neurons in the layer.
     */
    Dense(OutputVariant activationFunction, std::uint32_t units);
    /**
     * @brief add a dense layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param units the number of neurons in the layer.
     * @param norm the norm to use for regularization.
     */
    Dense(OutputVariant activationFunction, std::uint32_t units, NormVariant norm);
    ~Dense() = default;
    /**
     * @brief used to mark variables as input for the module.
     */
    void addInput(std::shared_ptr<Variable> input, std::uint32_t inputUnits) override
    {
        mpPaddingVariable->getInputs().push_back(input);
        
        // init default norm
        if(mpNorm == nullptr && mpsDefaultNorm != nullptr)
        {
            mpNorm = std::visit([](auto&& arg) {
                // Assuming all types in the variant can be dynamically casted to Operation*
                return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, *mpsDefaultNorm);
        }
        
        // +1 for the weight
        mpWeightMatrixVariable->getData() = std::make_shared<Tensor<double>>(Tensor<double>({inputUnits+1, mUnits})); // initialize the weights randomly
        
        if (mpNorm != nullptr) // adding norm to activation function
        {
            mpNormVariable = sGraph->addVariable(std::make_shared<Variable>(Variable(mpNorm, {mpWeightMatrixVariable}, {})));
            sGraph->addOutput(mpNormVariable);    
            mpWeightMatrixVariable->getConsumers().push_back(mpNormVariable);
        }
    }
    /**
     * @brief used to mark variables as output for the module.
     */
    void addOutput(std::shared_ptr<Variable> output) override
    {
        mpActivationVariable->getConsumers().push_back(output);
    }
    /**
     * @brief used to get the input variables of the module specified by the index.
     */
    std::shared_ptr<Variable> input(std::uint32_t index) override
    {
        return mpPaddingVariable;
    }
    /**
     * @brief used to get the output variables of the module specified by the index.
     */
    std::shared_ptr<Variable> output(std::uint32_t index) override
    {
        return mpActivationVariable;
    }
    /**
     * @brief used to set the default norm to use for regularization.
     */
    static void setDefaultNorm(NormVariant const & norm)
    {
        mpsDefaultNorm = std::make_shared<NormVariant>(norm);
    }
};

Dense::Dense(OutputVariant activationFunction, std::uint32_t units)
{
    // error checks
    if(sGraph == nullptr)
    {
        throw std::runtime_error("graph is not set");
    }
    mUnits = units; // set the number of neurons in the layer

    // create the variables

    mpPaddingVariable = sGraph->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Padding>(Padding(0,1,1)), {}, {}))); // pad for weights
    
    mpWeightMatrixVariable = sGraph->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {}))); // nullptr because there is no operation
    sLearnableParameters.push_back(mpWeightMatrixVariable);

    mpMatmulVariable = sGraph->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Matmul>(Matmul()), {mpPaddingVariable,mpWeightMatrixVariable}, {})));

    // turning the variant into a shared pointer to the operation class
    // Use std::visit to handle the variant
    std::shared_ptr<Operation> operation_ptr = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to Operation*
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, OutputVariant{activationFunction});

    mpActivationVariable = sGraph->addVariable(std::make_shared<Variable>(Variable(operation_ptr, {mpMatmulVariable}, {})));

    // conections within the module
    mpPaddingVariable->getConsumers().push_back(mpMatmulVariable);
    mpWeightMatrixVariable->getConsumers().push_back(mpMatmulVariable);
    
    mpMatmulVariable->getConsumers().push_back(mpActivationVariable);    

    
}

Dense::Dense(OutputVariant activationFunction, std::uint32_t units, NormVariant norm)
{
    mpNorm = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to Operation*
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, norm);
    Dense(activationFunction, units);
}

std::shared_ptr<NormVariant> Dense::mpsDefaultNorm = nullptr;