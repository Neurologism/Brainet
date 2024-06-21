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

/**
 * @brief class used to load the graph into the cluster class
 */
class __INIT__CLUSTER : public CLUSTER
{
public:
    __INIT__CLUSTER(GRAPH * graph)
    {
        __graph = graph;
    }
};

#endif // CLUSTER_INCLUDE_GUARD