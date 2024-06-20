#ifndef DENSE_INCLUDE_GUARD
#define DENSE_INCLUDE_GUARD

#include ".\cluster.h"

/**
 * @brief DENSE class creates a dense layer for the graph
 */
class DENSE : public CLUSTER
{
public:
    DENSE(OPERATION & input, int units, TENSOR & weight_matrix = TENSOR({0, 0}))
    {
        if(__graph == nullptr)
        {
            throw std::runtime_error("graph is not set");
        }
        __graph->add_variable(VARIABLE(nullptr, {}, {})); // nullptr because there is no operation
        VARIABLE * _weight_matrix_variable = &__graph->get_variables().back();
        __graph->add_variable(VARIABLE(new MATMUL(), {_weight_matrix_variable}, {}));
        VARIABLE * _matmul_variable = &__graph->get_variables().back();
    }
          
};

#endif // DENSE_INCLUDE_GUARD