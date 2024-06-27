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
protected:
    static GRAPH * __graph;
    int __units;
public:
    virtual void add_input(VARIABLE * input, int units){}
    virtual void add_output(VARIABLE * output){}
    virtual VARIABLE * input(int index = 0){return nullptr;}
    virtual VARIABLE * output(int index = 0){return nullptr;}
    static void set_graph(GRAPH * graph){__graph = graph;}
    int size(){return __units;}
};

GRAPH * CLUSTER::__graph = nullptr;

#include "input.h"
#include "dense.h"

using CLUSTER_VARIANT = std::variant<INPUT, DENSE>;

#endif // CLUSTER_INCLUDE_GUARD