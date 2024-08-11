#ifndef MODULE_HPP
#define MODULE_HPP

#include "../dependencies.hpp"
#include "../variable.hpp"
#include "../graph.hpp"

/**
 * @brief The Module class can be used to group multiple variables together. This is useful for adding substructures to the graph. It is mainly used to add something similar to layers to the graph.
 * It is intended that all cost functions add their variables to the graph in the constructor. 
 */
class Module
{
protected:
    static std::shared_ptr<Graph> __graph; // shared by all clusters , points to the graph into which the module should add its variables
    static std::vector<std::shared_ptr<Variable>> __learnable_parameters; // shared by all clusters , points to the learnable parameters of the graph for later use in the optimization process
    std::uint32_t __units = -1; // stores the input size of the module (could be moved to respective derived classes)
public:
    virtual ~Module() = default;
    /**
     * @brief virtual function that is supposed to be implemented by the derived classes. It is used to mark variables as input for the module. 
     */
    virtual void add_input(std::shared_ptr<Variable> input, std::uint32_t units){};
    /**
     * @brief virtual function that is supposed to be implemented by the derived classes. It is used to mark variables as output for the module.
     */
    virtual void add_output(std::shared_ptr<Variable> output){};
    /**
     * @brief virtual function that is supposed to be implemented by the derived classes. It is used to get the input variables of the module specified by the index.
     */
    virtual std::shared_ptr<Variable> input(std::uint32_t index = 0){return nullptr;}
    /**
     * @brief virtual function that is supposed to be implemented by the derived classes. It is used to get the output variables of the module specified by the index.
     */
    virtual std::shared_ptr<Variable> output(std::uint32_t index = 0){return nullptr;}
    /**
     * @brief wrapper class used to set the graph for all module objects manually. Manly intended for use with multiple graphs.
     */
    static void set_graph(std::shared_ptr<Graph> graph){__graph = graph;}
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
    static std::vector<std::shared_ptr<Variable>> & get_learnable_parameters(){return __learnable_parameters;}
};


// this is mainly for interface purposes

std::shared_ptr<Graph> Module::__graph = nullptr;
std::vector<std::shared_ptr<Variable>> Module::__learnable_parameters = {};


// code of all child classes
#include "input.hpp"
#include "dense.hpp"
#include "cost.hpp"

using Module_VARIANT = std::variant<Input, Dense, Cost>;

#endif // MODULE_HPP