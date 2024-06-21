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
    static GRAPH * __graph = nullptr;
public:
    virtual void add_input(VARIABLE * input){}
    virtual void add_output(VARIABLE * output){}
};

#endif // CLUSTER_INCLUDE_GUARD