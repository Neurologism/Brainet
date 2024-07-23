#ifndef CLUSTER_INCLUDE_GUARD
#define CLUSTER_INCLUDE_GUARD

#include "../dependencies.h"
#include "../variable.h"
#include "../graph.h"

/**
 * @brief The CLUSTER class can be used to group multiple variables together. This is useful for adding substructures to the graph. It is mainly used to add something similar to layers to the graph.
 */
class CLUSTER
{
protected:
    static std::shared_ptr<GRAPH> __graph; // shared by all clusters , points to the graph into which the cluster should add its variables
    static std::vector<std::shared_ptr<VARIABLE>> __learnable_parameters; // shared by all clusters , points to the learnable parameters of the graph for later use in the optimization process
    int __units = -1; // stores the input size of the cluster (could be moved to respective derived classes)
public:
    virtual ~CLUSTER() = default;
    /**
     * @brief virtual function that is supposed to be implemented by the derived classes. It is used to mark variables as input for the cluster. 
     */
    virtual void add_input(std::shared_ptr<VARIABLE> input, int units){};
    /**
     * @brief virtual function that is supposed to be implemented by the derived classes. It is used to mark variables as output for the cluster.
     */
    virtual void add_output(std::shared_ptr<VARIABLE> output){};
    /**
     * @brief virtual function that is supposed to be implemented by the derived classes. It is used to get the input variables of the cluster specified by the index.
     */
    virtual std::shared_ptr<VARIABLE> input(int index = 0){return nullptr;}
    /**
     * @brief virtual function that is supposed to be implemented by the derived classes. It is used to get the output variables of the cluster specified by the index.
     */
    virtual std::shared_ptr<VARIABLE> output(int index = 0){return nullptr;}
    /**
     * @brief wrapper class used to set the graph for all cluster objects manually. Manly intended for use with multiple graphs.
     */
    static void set_graph(std::shared_ptr<GRAPH> graph){__graph = graph;}
    /**
     * @brief used to get the private variable __units, could be moved to the respective derived classes.
     */
    int getUnits()
    {
        if(__units == -1)throw std::runtime_error("units not set");
        return __units;
    }
    static std::vector<std::shared_ptr<VARIABLE>> & get_learnable_parameters(){return __learnable_parameters;}
};

std::shared_ptr<GRAPH> CLUSTER::__graph = nullptr;
std::vector<std::shared_ptr<VARIABLE>> CLUSTER::__learnable_parameters = {};

#include "input.h"
#include "dense.h"
#include "cost.h"

using CLUSTER_VARIANT = std::variant<INPUT, DENSE, COST>;

#endif // CLUSTER_INCLUDE_GUARD