#ifndef DENSE_INCLUDE_GUARD
#define DENSE_INCLUDE_GUARD

#include ".\cluster.h"

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
    DENSE(OPERATION & activation_function, int units, TENSOR<double> & weight_matrix = TENSOR<double>({0, 0}))
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
        __graph->add_variable(VARIABLE(activation_function, {_matmul_variable}, {}));
        _activation_variable = &__graph->get_variables().back();
        _matmul_variable->get_consumers().push_back(_activation_variable);
        _weight_matrix_variable->get_consumers().push_back(_matmul_variable);
    }
    void build(VARIABLE * input, VARIABLE * output)
    {
        _matmul_variable->get_inputs().push_back(input);
        _activation_variable->get_consumers().push_back(output);
        _weight_matrix_variable->get_data() = TENSOR<double>({__units, input->get_shape()[0]}, 1);
    }
};

#endif // DENSE_INCLUDE_GUARD