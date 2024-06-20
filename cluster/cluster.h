#ifndef CLUSTER_INCLUDE_GUARD
#define CLUSTER_INCLUDE_GUARD

#include "../dependencies.h"
#include "../variable.h"

/**
 * @brief CLUSTER class is a wrapper class for a group of variables and operations.
 */
class CLUSTER
{
    std::vector<VARIABLE *> __variables, __inputs, __outputs;
    std::vector<OPERATION *> __operations;
public:
    std::vector<VARIABLE *> & get_variables() { return __variables; }
    std::vector<VARIABLE *> & get_inputs() { return __inputs; }
    std::vector<VARIABLE *> & get_outputs() { return __outputs; }
    std::vector<OPERATION *> & get_operations() { return __operations; }
};

#endif // CLUSTER_INCLUDE_GUARD