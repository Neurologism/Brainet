#ifndef DENSE_HPP
#define DENSE_HPP

#include "../operation/processing/dropout.hpp"
#include "../operation/matmul.hpp"
#include "../operation/processing/padding.hpp"
#include "../operation/activation_function/activation_function.hpp"
#include "../operation/parameter_norm_penalties/parameter_norm_penalty.hpp"
#include "../weight_initialization/weight_initializer.hpp"
#include "layer.hpp"

/**
 * @brief The dense module is intended for creating a dense (fully connected) layer in the graph.
 * It owns one input and one output variable.
 */
class Dense final : public Layer
{
    // storing index of the variables in the graph
    std::shared_ptr<Variable> mpWeightMatrixVariable; // learnable parameters of the layer (weights + bias)
    std::shared_ptr<Variable> mpMatmulVariable; // multiplication of the input and the weights
    std::shared_ptr<Variable> mpActivationVariable; // activation function applied
    std::shared_ptr<Variable> mpPaddingVariable; // used to pad the input with 1s for the bias
    std::shared_ptr<Variable> mpNormVariable; // used to compute a norm of the weights
    std::shared_ptr<Variable> mpDropoutVariable; // dropout applied to the input


    std::shared_ptr<WeightInitializer> mpWeightInitializer = std::make_shared<NormalizedInitialization>(); // used to initialize the weight matrix

    static std::shared_ptr<ParameterNormPenaltyVariant> mpsDefaultNorm; // default norm to use
    std::shared_ptr<Operation> mpNorm = nullptr; // norm to use for regularization



public:
    /**
     * @brief add a dense layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param units the number of neurons in the layer.
     * @param name the name of the module
     * @param dropout the dropout rate of the layer
     */
    Dense(const ActivationVariant &activationFunction, std::uint32_t units, const std::string& name = "", const double& dropout = 1.0);

    /**
     * @brief add a dense layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param size the number of neurons in the layer.
     * @param name the name of the module
     * @param dropout the dropout rate of the layer
     */
    Dense(const std::shared_ptr<Operation> &activationFunction, std::uint32_t size, const std::string& name = "", const double& dropout = 1.0);

    /**
     * @brief add a dense layer to the graph
     * @param activationFunction the operation representing the activation function.
     * @param size the number of neurons in the layer.
     * @param norm the norm to use for regularization.
     * @param name the name of the module
     * @param dropout the dropout rate of the layer
     */
    Dense(const ActivationVariant& activationFunction, std::uint32_t size, ParameterNormPenaltyVariant norm, const std::string& name = "", double dropout = 1.0);

    ~Dense() override = default;

    void addInput(const std::shared_ptr<Variable> &input, const std::uint32_t &inputSize) override;
    void addOutput(const std::shared_ptr<Variable> &output) override;

    std::vector<std::shared_ptr<Variable>> getInputs() override;
    std::vector<std::shared_ptr<Variable>> getOutputs() override;
    std::vector<std::shared_ptr<Variable>> getLearnableVariables() override;
    std::vector<std::shared_ptr<Variable>> getGradientVariables() override;

    void createWeightMatrix(std::uint32_t inputUnits);

    static void setDefaultNorm(ParameterNormPenaltyVariant const &norm);
};

inline Dense::Dense(const ActivationVariant &activationFunction, const std::uint32_t units, const std::string& name, const double& dropout) :
Dense(std::visit([]<typename T0>(T0&& arg) {
    // Assuming all types in the variant can be dynamically cast to Operation*
    return std::shared_ptr<Operation>(std::make_shared<std::decay_t<T0>>(arg));}, activationFunction), units, name, dropout)
{

}


inline Dense::Dense(const std::shared_ptr<Operation> &activationFunction, const std::uint32_t size, const std::string& name, const double& dropout) : Layer(name)
{
    mSize = size; // set the number of neurons in the layer

    // create the variables
    mpDropoutVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Dropout>(Dropout(dropout)))));
    mpPaddingVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Padding>(Padding(0,1,1)), {mpDropoutVariable}))); // pad for weights
    mpWeightMatrixVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr))); // nullptr because there is no operation
    mpMatmulVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Matmul>(Matmul()), {mpPaddingVariable,mpWeightMatrixVariable})));
    mpActivationVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(activationFunction, {mpMatmulVariable})));

    // connections within the module
    mpDropoutVariable->getConsumers().push_back(mpPaddingVariable);
    mpPaddingVariable->getConsumers().push_back(mpMatmulVariable);
    mpWeightMatrixVariable->getConsumers().push_back(mpMatmulVariable);
    mpMatmulVariable->getConsumers().push_back(mpActivationVariable);
   
}

inline Dense::Dense(const ActivationVariant& activationFunction, const std::uint32_t size, ParameterNormPenaltyVariant norm, const std::string& name, const double dropout) : Dense(activationFunction, units, name, dropout)
{
    mpNorm = std::visit([]<typename T0>(T0&& arg) {
        // Assuming all types in the variant can be dynamically cast to Operation*
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<T0>>(arg));}, norm);
    
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

inline std::vector<std::shared_ptr<Variable>> Dense::getInputs()
{
    return {mpDropoutVariable};
}

inline std::vector<std::shared_ptr<Variable>> Dense::getOutputs()
{
    return {mpActivationVariable};
}

inline std::vector<std::shared_ptr<Variable>> Dense::getLearnableVariables()
{
    if (mpNorm != nullptr)
    {
        return {mpWeightMatrixVariable, mpNormVariable};
    }
    return {mpWeightMatrixVariable};
}

inline std::vector<std::shared_ptr<Variable>> Dense::getGradientVariables()
{
    if (mpNorm != nullptr)
    {
        return {mpNormVariable};
    }
    return {};
}

inline void Dense::createWeightMatrix(std::uint32_t inputUnits)
{
    mpWeightMatrixVariable->getData() = std::make_shared<Tensor<double>>(Tensor<double>({inputUnits+1, mSize})); // initialize the weights randomly

    // initialize the weights randomly
    mpWeightInitializer->createRandomEngine(inputUnits, mSize);
    std::vector<double> weights = mpWeightInitializer->createRandomVector();

    for (std::uint32_t i = 0; i < inputUnits; i++) // load the weights into the weight matrix
    {
        for (std::uint32_t j = 0; j < mSize; j++)
        {
            mpWeightMatrixVariable->getData()->set({i, j}, weights[i*mSize+j]);
        }
    }

    // initialize the bias
    double bias = 0.0;

    std::type_info const & type = typeid(mpActivationVariable->getOperation());
    if (type == typeid(ReLU))
    {
        bias = 0.1;
    }

    for (std::uint32_t j = 0; j < mSize; j++)
    {
        mpWeightMatrixVariable->getData()->set({inputUnits, j}, bias);
    }
}

inline void Dense::setDefaultNorm(ParameterNormPenaltyVariant const & norm)
{
    mpsDefaultNorm = std::make_shared<ParameterNormPenaltyVariant>(norm);
}


std::shared_ptr<ParameterNormPenaltyVariant> Dense::mpsDefaultNorm = nullptr;

#endif // DENSE_HPP