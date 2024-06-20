#ifndef CLUSTER_INCLUDE_GUARD
#define CLUSTER_INCLUDE_GUARD

#include "../dependencies.h"
#include "../variable.h"
#include "../graph.h"

/**
 * @brief CLUSTER class is a wrapper class for a group of variables and operations.
 */
class CLUSTER
{
    std::vector<VARIABLE *> __variables, __inputs, __outputs;
    std::vector<OPERATION *> __operations;
    static GRAPH * __graph = nullptr;
public:
    // constructor should be called to set graph
    CLUSTER(GRAPH * graph) { __graph = graph; }
    std::vector<VARIABLE *> & get_variables() { return __variables; }
    std::vector<VARIABLE *> & get_inputs() { return __inputs; }
    std::vector<VARIABLE *> & get_outputs() { return __outputs; }
    std::vector<OPERATION *> & get_operations() { return __operations; }
};

#endif // CLUSTER_INCLUDE_GUARD