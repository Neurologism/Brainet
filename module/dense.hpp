#ifndef DENSE_HPP
#define DENSE_HPP

#include "../operation/processing/dropout.hpp"
#include "../operation/matmul.hpp"
#include "../operation/processing/padding.hpp"
#include "../operation/activation_function/activation_function.hpp"
#include "../operation/parameter_norm_penalties/parameter_norm_penalty.hpp"
#include "../weight_initialization/weight_initializer.hpp"

/**
 * @brief The dense module is intended for creating a dense (fully connected) layer in the graph.
 * It owns one input and one output variable.
 */
class Dense final : public Module
{
    // storing index of the variables in the graph
    std::shared_ptr<Variable> mpWeightMatrixVariable; // learnable parameters of the layer (weights + bias)
    std::shared_ptr<Variable> mpMatmulVariable; // multiplication of the input and the weights
    std::shared_ptr<Variable> mpActivationVariable; // activation function applied
    std::shared_ptr<Variable> mpPaddingVariable; // used to pad the input with 1s for the bias
    std::shared_ptr<Variable> mpNormVariable; // used to compute a norm of the weights

    std::shared_ptr<WeightInitializer> mpWeightInitializer = std::make_shared<NormalizedInitialization>(); // weight initializer

    static std::shared_ptr<ParameterNormPenaltyVariant> mpsDefaultNorm; // default norm to use
    std::shared_ptr<Operation> mpNorm = nullptr; // norm to use for regularization

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

    void createWeightMatrix(std::uint32_t inputUnits);

    static void setDefaultNorm(ParameterNormPenaltyVariant const &norm);
};

inline Dense::Dense(const HiddenVariant &activationFunction, const std::uint32_t units, const std::string& name, const double& dropout) :
Dense(std::visit([]<typename T0>(T0&& arg) {
    // Assuming all types in the variant can be dynamically cast to Operation*
    return std::shared_ptr<Operation>(std::make_shared<std::decay_t<T0>>(arg));}, activationFunction), units, name, dropout)
{

}


inline Dense::Dense(const std::shared_ptr<Operation> &activationFunction, const std::uint32_t units, const std::string& name, const double& dropout) : Module(name)
{
    mUnits = units; // set the number of neurons in the layer

    // create the variables
    mpPaddingVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Padding>(Padding(0,1,1))))); // pad for weights
    mpWeightMatrixVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {}))); // nullptr because there is no operation
    mpMatmulVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Matmul>(Matmul()), {mpPaddingVariable,mpWeightMatrixVariable})));
    mpActivationVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(activationFunction, {mpMatmulVariable})));
    
    mpDropoutVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Dropout>(Dropout(dropout)), {mpActivationVariable})));

    // connections within the module
    mpPaddingVariable->getConsumers().push_back(mpMatmulVariable);
    mpWeightMatrixVariable->getConsumers().push_back(mpMatmulVariable);
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

inline void Dense::createWeightMatrix(std::uint32_t inputUnits)
{
    mpWeightMatrixVariable->getData() = std::make_shared<Tensor<double>>(Tensor<double>({inputUnits+1, mUnits})); // initialize the weights randomly

    // initialize the weights randomly
    mpWeightInitializer->createRandomEngine(inputUnits, mUnits);
    std::vector<double> weights = mpWeightInitializer->createRandomVector();

    for (std::uint32_t i = 0; i < inputUnits; i++) // load the weights into the weight matrix
    {
        for (std::uint32_t j = 0; j < mUnits; j++)
        {
            mpWeightMatrixVariable->getData()->set({i, j}, weights[i*mUnits+j]);
        }
    }

    // initialize the bias
    double bias = 0.0;

    std::type_info const & type = typeid(mpActivationVariable->getOperation());
    if (type == typeid(ReLU))
    {
        bias = 0.1;
    }

    for (std::uint32_t j = 0; j < mUnits; j++)
    {
        mpWeightMatrixVariable->getData()->set({inputUnits, j}, bias);
    }
}

inline void Dense::setDefaultNorm(ParameterNormPenaltyVariant const & norm)
{
    mpsDefaultNorm = std::make_shared<ParameterNormPenaltyVariant>(norm);
}


std::shared_ptr<ParameterNormPenaltyVariant> FullyConnected::mpsDefaultNorm = nullptr;

#endif // DENSE_HPP