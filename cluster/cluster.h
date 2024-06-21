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
    // constructor should be called to set graph
    CLUSTER(GRAPH * graph) { __graph = graph; }
};

#endif // CLUSTER_INCLUDE_GUARD