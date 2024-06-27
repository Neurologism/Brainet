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
    // storing index of the variables in the graph
    int _weight_matrix_variable;
    int _matmul_variable;
    int _activation_variable;

public:
    DENSE(ACTIVATION_FUNCTION_VARIANT activation_function, int units, TENSOR<double> weight_matrix = TENSOR<double>({0, 0}));
    void add_input(VARIABLE * input, int units) override
    {
        __graph->at(_matmul_variable)->get_inputs()->push_back(input);
        *(__graph->at(_weight_matrix_variable)->get_data()) = TENSOR<double>({__units, units}, 1);
    }
    void add_output(VARIABLE * output) override
    {
        __graph->at(_activation_variable)->get_consumers()->push_back(output);
    }
    VARIABLE * input(int index) override
    {
        return __graph->at(_matmul_variable);
    }
    VARIABLE * output(int index) override
    {
        return __graph->at(_activation_variable);
    }
};

DENSE::DENSE(ACTIVATION_FUNCTION_VARIANT activation_function, int units, TENSOR<double> weight_matrix)
{
    if(__graph == nullptr)
    {
        throw std::runtime_error("graph is not set");
    }
    __units = units;

    // create the variables
    _weight_matrix_variable = __graph->get_variables().size();
    __graph->add_variable(VARIABLE(nullptr, {}, {})); // nullptr because there is no operation

    _matmul_variable = __graph->get_variables().size();
    __graph->add_variable(VARIABLE(new MATMUL(), {__graph->at(_weight_matrix_variable)}, {}));

    // Use std::visit to handle the variant
    auto operation_ptr = std::visit([](auto&& arg) -> OPERATION* {
        // Assuming all types in the variant can be dynamically casted to OPERATION*
        return dynamic_cast<OPERATION*>(&arg);
    }, activation_function);

    _activation_variable = __graph->get_variables().size();
    __graph->add_variable(VARIABLE(operation_ptr, {__graph->at(_matmul_variable)}, {}));

    // conections within the cluster
    __graph->at(_weight_matrix_variable)->get_consumers()->push_back(__graph->at(_matmul_variable));
    __graph->at(_matmul_variable)->get_consumers()->push_back(__graph->at(_activation_variable));
    
}

#endif // DENSE_INCLUDE_GUARD