#ifndef DENSE_INCLUDE_GUARD
#define DENSE_INCLUDE_GUARD

#include ".\cluster.h"
#include "..\operation\linear_algebra\matmul.h"
#include "..\operation\activation_function\activation_function.h"

/**
 * @brief DENSE class creates a dense layer for the graph
 */
class DENSE : public CLUSTER
{
    int __units;
    VARIABLE * _weight_matrix_variable;
    VARIABLE * _matmul_variable;
    VARIABLE * _activation_variable;

public:
    DENSE(ACTIVATION_FUNCTION_VARIANT activation_function, int units, TENSOR<double> weight_matrix = TENSOR<double>({0, 0}));
    void add_input(VARIABLE * input) override
    {
        _matmul_variable->get_inputs().push_back(input);
        _weight_matrix_variable->get_data() = TENSOR<double>({__units, input->get_data().shape(0)}, 1);
    }
    void add_output(VARIABLE * output) override
    {
        _activation_variable->get_consumers().push_back(output);
    }
    VARIABLE * input(int index) override
    {
        return _matmul_variable;
    }
    VARIABLE * output(int index) override
    {
        return _activation_variable;
    }
};

DENSE::DENSE(ACTIVATION_FUNCTION_VARIANT activation_function, int units, TENSOR<double> weight_matrix)
{
    if(__graph == nullptr)
    {
        throw std::runtime_error("graph is not set");
    }
    __units = units;
    __graph->add_variable(VARIABLE(nullptr, {}, {})); // nullptr because there is no operation
    _weight_matrix_variable = &__graph->get_variables().back();
    __graph->add_variable(VARIABLE(new MATMUL(), {_weight_matrix_variable}, {}));
    _matmul_variable = &__graph->get_variables().back();
    // Use std::visit to handle the variant
    auto operation_ptr = std::visit([](auto&& arg) -> OPERATION* {
        // Assuming all types in the variant can be dynamically casted to OPERATION*
        return dynamic_cast<OPERATION*>(&arg);
    }, activation_function);

    __graph->add_variable(VARIABLE(operation_ptr, {_matmul_variable}, {}));
    _activation_variable = &__graph->get_variables().back();
    _matmul_variable->get_consumers().push_back(_activation_variable);
    _weight_matrix_variable->get_consumers().push_back(_matmul_variable);
}

#endif // DENSE_INCLUDE_GUARD