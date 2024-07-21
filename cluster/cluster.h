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
    static std::shared_ptr<GRAPH> __graph;
    static std::vector<std::shared_ptr<VARIABLE>> __learnable_parameters;
    int __units = -1;
public:
    virtual ~CLUSTER() = default;
    virtual void add_input(std::shared_ptr<VARIABLE> input, int units){}
    virtual void add_output(std::shared_ptr<VARIABLE> output){}
    virtual std::shared_ptr<VARIABLE> input(int index = 0){return nullptr;}
    virtual std::shared_ptr<VARIABLE> output(int index = 0){return nullptr;}
    static void set_graph(std::shared_ptr<GRAPH> graph){__graph = graph;}
    int size()
    {
        if(__units == -1)throw std::runtime_error("units not set");
        return __units;
    }
    int getUnits(){return __units;}
    static std::vector<std::shared_ptr<VARIABLE>> get_learnable_parameters(){return __learnable_parameters;}
};

std::shared_ptr<GRAPH> CLUSTER::__graph = nullptr;
std::vector<std::shared_ptr<VARIABLE>> CLUSTER::__learnable_parameters = {};

#include "input.h"
#include "dense.h"
#include "cost.h"

using CLUSTER_VARIANT = std::variant<INPUT, DENSE, COST>;

#endif // CLUSTER_INCLUDE_GUARD