#ifndef CLUSTER_INCLUDE_GUARD
#define CLUSTER_INCLUDE_GUARD

#include "../dependencies.h"
#include "../variable.h"
#include "../graph.h"

/**
 * @brief The MODULE class can be used to group multiple variables together. This is useful for adding substructures to the graph. It is mainly used to add something similar to layers to the graph.
 * It is intended that all cost functions add their variables to the graph in the constructor. 
 */
class MODULE
{
protected:
    static std::shared_ptr<GRAPH> __graph; // shared by all clusters , points to the graph into which the module should add its variables
    static std::vector<std::shared_ptr<VARIABLE>> __learnable_parameters; // shared by all clusters , points to the learnable parameters of the graph for later use in the optimization process
    std::uint32_t __units = -1; // stores the input size of the module (could be moved to respective derived classes)
public:
    virtual ~MODULE() = default;
    /**
     * @brief virtual function that is supposed to be implemented by the derived classes. It is used to mark variables as input for the module. 
     */
    virtual void add_input(std::shared_ptr<VARIABLE> input, std::uint32_t units){};
    /**
     * @brief virtual function that is supposed to be implemented by the derived classes. It is used to mark variables as output for the module.
     */
    virtual void add_output(std::shared_ptr<VARIABLE> output){};
    /**
     * @brief virtual function that is supposed to be implemented by the derived classes. It is used to get the input variables of the module specified by the index.
     */
    virtual std::shared_ptr<VARIABLE> input(std::uint32_t index = 0){return nullptr;}
    /**
     * @brief virtual function that is supposed to be implemented by the derived classes. It is used to get the output variables of the module specified by the index.
     */
    virtual std::shared_ptr<VARIABLE> output(std::uint32_t index = 0){return nullptr;}
    /**
     * @brief wrapper class used to set the graph for all module objects manually. Manly intended for use with multiple graphs.
     */
    static void set_graph(std::shared_ptr<GRAPH> graph){__graph = graph;}
    /**
     * @brief used to get the private variable __units, could be moved to the respective derived classes.
     */
    std::uint32_t getUnits()
    {
        if(__units == -1)throw std::runtime_error("units not set");
        return __units;
    }
    /**
     * @brief used to get all learnable parameters declared by the module objects.
     */
    static std::vector<std::shared_ptr<VARIABLE>> & get_learnable_parameters(){return __learnable_parameters;}
};


// this is mainly for interface purposes

std::shared_ptr<GRAPH> MODULE::__graph = nullptr;
std::vector<std::shared_ptr<VARIABLE>> MODULE::__learnable_parameters = {};


// code of all child classes
#include "input.h"
#include "dense.h"
#include "cost.h"

using MODULE_VARIANT = std::variant<INPUT, DENSE, COST>;

#endif // MODULE_INCLUDE_GUARD