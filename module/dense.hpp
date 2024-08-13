#ifndef DENSE_HPP
#define DENSE_HPP

#include "./module.hpp"
#include "../operation/matmul.hpp"
#include "../operation/processing/padding.hpp"
#include "../operation/activation_function/activation_function.hpp"
#include "../operation/norm/norm.hpp"

/**
 * @brief the dense module is intended for creating a dense (fully connected) layer in the graph. It owns 1 input and 1 output variable.
 */
class Dense : public Module
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
    Dense(ActivationVariant activationFunction, std::uint32_t units);
    /**
     * @brief add a dense layer to the graph
     * @param activation_function the operation representing the activation function.
     * @param units the number of neurons in the layer.
     * @param norm the norm to use for regularization.
     */
    Dense(ActivationVariant activation_function, std::uint32_t units, NormVariant norm);
    ~Dense() = default;
    /**
     * @brief used to mark variables as input for the module.
     */
    void add_input(std::shared_ptr<Variable> input, std::uint32_t input_units) override
    {
        mpPaddingVariable->get_inputs().push_back(input);
        
        // init default norm
        if(mpNorm == nullptr && mpsDefaultNorm != nullptr)
        {
            mpNorm = std::visit([](auto&& arg) {
                // Assuming all types in the variant can be dynamically casted to Operation*
                return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, *mpsDefaultNorm);
        }
        
        // +1 for the weight
        mpWeightMatrixVariable->get_data() = std::make_shared<Tensor<double>>(Tensor<double>({input_units+1,__units})); // initialize the weights randomly
        
        if (mpNorm != nullptr) // adding norm to activation function
        {
            mpNormVariable = sGraph->add_variable(std::make_shared<Variable>(Variable(mpNorm, {mpWeightMatrixVariable}, {})));
            sGraph->add_output(mpNormVariable);    
            mpWeightMatrixVariable->get_consumers().push_back(mpNormVariable);
        }
    }
    /**
     * @brief used to mark variables as output for the module.
     */
    void add_output(std::shared_ptr<Variable> output) override
    {
        mpActivationVariable->get_consumers().push_back(output);
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
    static void setmpsDefaultNorm(NormVariant const & norm)
    {
        mpsDefaultNorm = std::make_shared<NormVariant>(norm);
    }
};

Dense::Dense(ActivationVariant activation_function, std::uint32_t units)
{
    // error checks
    if(sGraph == nullptr)
    {
        throw std::runtime_error("graph is not set");
    }
    __units = units; // set the number of neurons in the layer

    // create the variables

    mpPaddingVariable = sGraph->add_variable(std::make_shared<Variable>(Variable(std::make_shared<Padding>(Padding(0,1,1)), {}, {}))); // pad for weights
    
    mpWeightMatrixVariable = sGraph->add_variable(std::make_shared<Variable>(Variable(nullptr, {}, {}))); // nullptr because there is no operation
    __learnable_parameters.push_back(mpWeightMatrixVariable);

    mpMatmulVariable = sGraph->add_variable(std::make_shared<Variable>(Variable(std::make_shared<Matmul>(Matmul()), {mpPaddingVariable,mpWeightMatrixVariable}, {})));

    // turning the variant into a shared pointer to the operation class
    // Use std::visit to handle the variant
    std::shared_ptr<Operation> operation_ptr = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to Operation*
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, ActivationVariant{activation_function});

    mpActivationVariable = sGraph->add_variable(std::make_shared<Variable>(Variable(operation_ptr, {mpMatmulVariable}, {})));

    // conections within the module
    mpPaddingVariable->get_consumers().push_back(mpMatmulVariable);
    mpWeightMatrixVariable->get_consumers().push_back(mpMatmulVariable);
    
    mpMatmulVariable->get_consumers().push_back(mpActivationVariable);    

    
}

Dense::Dense(ActivationVariant activation_function, std::uint32_t units, NormVariant norm)
{
    mpNorm = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to Operation*
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, norm);
    Dense(activation_function, units);
}

std::shared_ptr<NormVariant> Dense::mpsDefaultNorm = nullptr;

#endif // DENSE_HPP