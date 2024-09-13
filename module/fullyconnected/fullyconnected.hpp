#ifndef FULLYCONNECTED_HPP
#define FULLYCONNECTED_HPP

#include "../../operation/matmul.hpp"
#include "../../operation/processing/padding.hpp"
#include "../../operation/activation_function/activation_function.hpp"
#include "../../operation/parameter_norm_penalties/parameter_norm_penalty.hpp"
#include "../../weight_initialization/weight_initializer.hpp"
#include "../module.hpp"

/**
 * @brief The fully connected module is intended
 for creating a fully connected layer without activation function in the graph.
 It's the base class for the dense and output modules.
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

    std::shared_ptr<WeightInitializer> mpWeightInitializer = std::make_shared<NormalizedInitialization>(); // weight initializer

    static std::shared_ptr<ParameterNormPenaltyVariant> mpsDefaultNorm; // default norm to use
    std::shared_ptr<Operation> mpNorm = nullptr; // norm to use for regularization

    void addInput(const std::shared_ptr<Variable> &input, const std::uint32_t &inputSize) override;
    void addOutput(const std::shared_ptr<Variable> &output) override;
public:
    /**
     * @brief add a fully connected layer to the graph
     * @param units the number of neurons in the layer
     * @param name the name of the module
     */
    explicit FullyConnected( std::uint32_t units, const std::string &name = "" );

    /**
     * @brief initialize the weight matrix of the layer with random values
     * @param inputUnits the number of weights each neuron has
     */
    void createWeightMatrix(std::uint32_t inputUnits);

    /**
     * @brief Set the default norm to use for regularization.
     * Layers are initialized with this norm if no other norm is specified.
     * @param norm the norm to use
     */

};

inline FullyConnected::FullyConnected(const std::uint32_t units, const std::string &name) : Module(name)
{

}

#endif // FULLYCONNECTED_HPP