#ifndef INPUT_INCLUDE_GUARD
#define INPUT_INCLUDE_GUARD

#include ".\cluster.h"

/**
 * @brief INPUT class creates an input variable for the graph
 * it is a wrapper class for a variable that is not the result of an operation
 */
class INPUT : public CLUSTER
{
public:
    INPUT(TENSOR & data, int units)
    {
        if(__graph == nullptr)
        {
            throw std::runtime_error("graph is not set");
        }
        __graph->add_variable(VARIABLE(nullptr, {}, {}, data));
    }
};

#endif // INPUT_INCLUDE_GUARD