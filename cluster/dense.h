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
    std::shared_ptr<VARIABLE> _weight_matrix_variable;
    std::shared_ptr<VARIABLE> _matmul_variable;
    std::shared_ptr<VARIABLE> _activation_variable;

public:
    DENSE(ACTIVATION_FUNCTION_VARIANT activation_function, int units);
    ~DENSE() = default;
    void add_input(std::shared_ptr<VARIABLE> input, int units) override
    {
        _activation_variable->get_inputs().push_back(input);
    }
    void add_output(std::shared_ptr<VARIABLE> output) override
    {
        _activation_variable->get_consumers().push_back(output);
    }
    std::shared_ptr<VARIABLE> input(int index) override
    {
        return _activation_variable;
    }
    std::shared_ptr<VARIABLE> output(int index) override
    {
        return _activation_variable;
    }
};

DENSE::DENSE(ACTIVATION_FUNCTION_VARIANT activation_function, int units)
{
    if(__graph == nullptr)
    {
        throw std::runtime_error("graph is not set");
    }
    __units = units;

    // // create the variables
    
    // _weight_matrix_variable = __graph->add_variable(VARIABLE(nullptr, {}, {})); // nullptr because there is no operation

    // _matmul_variable = __graph->add_variable(VARIABLE(new MATMUL(), {_weight_matrix_variable}, {}));

    // Use std::visit to handle the variant
    std::shared_ptr<OPERATION> operation_ptr = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to OPERATION*
        return std::shared_ptr<OPERATION>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, ACTIVATION_FUNCTION_VARIANT{activation_function});

    _activation_variable = __graph->add_variable(std::make_shared<VARIABLE>(VARIABLE(operation_ptr, {}, {})));

    // conections within the cluster
    // _weight_matrix_variable->get_consumers()->push_back(_matmul_variable);
    // _matmul_variable->get_consumers()->push_back(_activation_variable);    
}

#endif // DENSE_INCLUDE_GUARD