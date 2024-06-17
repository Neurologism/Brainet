#ifndef BUILDER_INCLUDE_GUARD
#define BUILDER_INCLUDE_GUARD

#include "../graph.h"
#include "../variable.h"
#include "../dependencies.h"

/**
 * @brief BUILDER class is a builder class for turning a model in a graph out of operations.
*/

class BUILDER
{
protected:
    GRAPH * __graph;
public:
    BUILDER(GRAPH * graph) : __graph(graph){};
};
#endif // BUILDER_INCLUDE_GUARD