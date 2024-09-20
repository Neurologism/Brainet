#ifndef DENSE_HPP
#define DENSE_HPP

#include "../operation/processing/dropout.hpp"
#include "../operation/matmul.hpp"
#include "../operation/processing/padding.hpp"
#include "../operation/activation_function/activation_function.hpp"
#include "../operation/parameter_norm_penalties/parameter_norm_penalty.hpp"
#include "../operation/weight_initialization/weight_matrix_initializer.hpp"
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

    std::vector<std::shared_ptr<Variable>> getInputs() override;
    std::vector<std::shared_ptr<Variable>> getOutputs() override;
    std::vector<std::shared_ptr<Variable>> getLearnableVariables() override;
    std::vector<std::shared_ptr<Variable>> getGradientVariables() override;

    void createWeightMatrix(std::uint32_t n);

    static void setDefaultNorm(ParameterNormPenaltyVariant const &norm);
};

#endif // DENSE_HPP