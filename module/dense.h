#ifndef DENSE_INCLUDE_GUARD
#define DENSE_INCLUDE_GUARD

#include "./module.h"
#include "../operation/linear_algebra/matmul.h"
#include "../operation/processing/padding.h"
#include "../operation/activation_function/activation_function.h"

/**
 * @brief the dense module is intended for creating a dense (fully connected) layer in the graph. It owns 1 input and 1 output variable.
 */
class DENSE : public MODULE
{
    // storing index of the variables in the graph
    std::shared_ptr<VARIABLE> _weight_matrix_variable; // learnable parameters of the layer (weights + bias)
    std::shared_ptr<VARIABLE> _matmul_variable; // multiplication of the input and the weights
    std::shared_ptr<VARIABLE> _activation_variable; // activation function applied
    std::shared_ptr<VARIABLE> _padding_variable; // used to pad the input with 1s for the bias

public:
    /**
     * @brief add a dense layer to the graph
     * @param activation_function the operation representing the activation function.
     * @param units the number of neurons in the layer.
     */
    DENSE(ACTIVATION_FUNCTION_VARIANT activation_function, std::uint32_t units);
    ~DENSE() = default;
    /**
     * @brief used to mark variables as input for the module.
     */
    void add_input(std::shared_ptr<VARIABLE> input, std::uint32_t input_units) override
    {
        _padding_variable->get_inputs().push_back(input);
        _weight_matrix_variable->get_data() = std::make_shared<TENSOR<double>>(TENSOR<double>({input_units+1,__units}, 1, 1)); // we now know the size of the input (make own function for better use maybe)
    }
    /**
     * @brief used to mark variables as output for the module.
     */
    void add_output(std::shared_ptr<VARIABLE> output) override
    {
        _activation_variable->get_consumers().push_back(output);
    }
    /**
     * @brief used to get the input variables of the module specified by the index.
     */
    std::shared_ptr<VARIABLE> input(std::uint32_t index) override
    {
        return _padding_variable;
    }
    /**
     * @brief used to get the output variables of the module specified by the index.
     */
    std::shared_ptr<VARIABLE> output(std::uint32_t index) override
    {
        return _activation_variable;
    }
};

DENSE::DENSE(ACTIVATION_FUNCTION_VARIANT activation_function, std::uint32_t units)
{
    // error checks
    if(__graph == nullptr)
    {
        throw std::runtime_error("graph is not set");
    }
    __units = units; // set the number of neurons in the layer

    // create the variables

    _padding_variable = __graph->add_variable(std::make_shared<VARIABLE>(VARIABLE(std::make_shared<Padding>(Padding(0,1,1)), {}, {}))); // pad for weights
    
    _weight_matrix_variable = __graph->add_variable(std::make_shared<VARIABLE>(VARIABLE(nullptr, {}, {}))); // nullptr because there is no operation
    __learnable_parameters.push_back(_weight_matrix_variable);

    _matmul_variable = __graph->add_variable(std::make_shared<VARIABLE>(VARIABLE(std::make_shared<MATMUL>(MATMUL()), {_padding_variable,_weight_matrix_variable}, {})));

    // turning the variant into a shared pointer to the operation class
    // Use std::visit to handle the variant
    std::shared_ptr<OPERATION> operation_ptr = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to OPERATION*
        return std::shared_ptr<OPERATION>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, ACTIVATION_FUNCTION_VARIANT{activation_function});

    _activation_variable = __graph->add_variable(std::make_shared<VARIABLE>(VARIABLE(operation_ptr, {_matmul_variable}, {})));

    // conections within the module
    _padding_variable->get_consumers().push_back(_matmul_variable);
    _weight_matrix_variable->get_consumers().push_back(_matmul_variable);
    _matmul_variable->get_consumers().push_back(_activation_variable);    
}

#endif // DENSE_INCLUDE_GUARD