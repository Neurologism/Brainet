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
public:
    virtual void add_input(VARIABLE * input){}
    virtual void add_output(VARIABLE * output){}
    virtual VARIABLE * input(int index = 0){return nullptr;}
    virtual VARIABLE * output(int index = 0){return nullptr;}
};

#endif // CLUSTER_INCLUDE_GUARD