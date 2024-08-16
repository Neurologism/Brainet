#ifndef FULLYCONNECTED_HPP
#define FULLYCONNECTED_HPP

#include "../module.hpp"
#include "../../operation/matmul.hpp"
#include "../../operation/processing/padding.hpp"
#include "../../operation/activation_function/activation_function.hpp"
#include "../../operation/norm/norm.hpp"

/**
 * @brief the fully connected module is intended for creating a fully connected layer without activation function in the graph. It's the base class for the dense and output modules.
 */
class FullyConnected : public Module
{
protected:
    // storing index of the variables in the graph
    std::shared_ptr<Variable> mpWeightMatrixVariable; // learnable parameters of the layer (weights + bias)
    std::shared_ptr<Variable> mpMatmulVariable; // multiplication of the input and the weights
    std::shared_ptr<Variable> mpActivationVariable; // activation function applied
    std::shared_ptr<Variable> mpPaddingVariable; // used to pad the input with 1s for the bias
    std::shared_ptr<Variable> mpNormVariable; // used to compute a norm of the weights

    static std::shared_ptr<NormVariant> mpsDefaultNorm; // default norm to use
    std::shared_ptr<Operation> mpNorm = nullptr; // norm to use for regularization

public:

    FullyConnected( std::uint32_t units);
    


    /**
     * @brief used to initialize the module with the input and output variables.
     */
    void __init__(std::vector<std::shared_ptr<Variable>> initialInpus, std::vector<std::shared_ptr<Variable>> initialOutputs) override;

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

    /**
     * @brief used to create the weight matrix for the dense layer.
     */
    void createWeightMatrix(std::uint32_t inputUnits);

    /**
     * @brief used to set the default norm to use for regularization.
     */
    static void setDefaultNorm(NormVariant const & norm)
    {
        mpsDefaultNorm = std::make_shared<NormVariant>(norm);
    }

};

FullyConnected::FullyConnected(std::uint32_t units)
{
    mUnits = units; // set the number of neurons in the layer

    // create the variables

    mpPaddingVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Padding>(Padding(0,1,1)), {}, {}))); // pad for weights
    
    mpWeightMatrixVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {}))); // nullptr because there is no operation

    mpMatmulVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Matmul>(Matmul()), {mpPaddingVariable,mpWeightMatrixVariable}, {})));


    // conections within the module
    mpPaddingVariable->getConsumers().push_back(mpMatmulVariable);
    mpWeightMatrixVariable->getConsumers().push_back(mpMatmulVariable);
}

void FullyConnected::__init__(std::vector<std::shared_ptr<Variable>> initialInpus, std::vector<std::shared_ptr<Variable>> initialOutputs)
{
    if (initialInpus.size() != 1)
    {
        throw std::invalid_argument("FullyConnected::__init__: the number of input variables must be 1");
    }
    if (initialOutputs.size() != 1)
    {
        throw std::invalid_argument("FullyConnected::__init__: the number of output variables must be 1");
    }

    mpPaddingVariable->getInputs().push_back(initialInpus[0]);
        
    // init default norm
    if(mpNorm == nullptr && mpsDefaultNorm != nullptr)
    {
        mpNorm = std::visit([](auto&& arg) {
            // Assuming all types in the variant can be dynamically casted to Operation*
            return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, *mpsDefaultNorm);
    }
    
    if (mpNorm != nullptr) // adding norm to activation function
    {
        mpNormVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(mpNorm, {mpWeightMatrixVariable}, {})));  
        mpWeightMatrixVariable->getConsumers().push_back(mpNormVariable);
    }


    mpActivationVariable->getConsumers().push_back(initialOutputs[0]);
}

std::shared_ptr<Variable> FullyConnected::getVariable(std::uint32_t index)
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

void FullyConnected::createWeightMatrix(std::uint32_t inputUnits)
{
    mpWeightMatrixVariable->getData() = std::make_shared<Tensor<double>>(Tensor<double>({inputUnits+1, mUnits})); // initialize the weights randomly

    // initialize the weights randomly
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-std::sqrt(2/(inputUnits+mUnits)), std::sqrt(2/(inputUnits+mUnits)));
    for (std::uint32_t i = 0; i < inputUnits; i++) // dont initialize the bias
    {
        for (std::uint32_t j = 0; j < mUnits; j++)
        {
            mpWeightMatrixVariable->getData()->set({i,j},dis(gen));
        }
    }

}


std::shared_ptr<NormVariant> FullyConnected::mpsDefaultNorm = nullptr;

#endif // FULLYCONNECTED_HPP