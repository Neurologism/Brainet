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
    std::uint32_t mUnits = -1; // stores the input size of the module (could be moved to respective derived classes)
public:
    virtual ~Module() = default;
    /**
     * @brief virtual function that is supposed to be implemented by the derived classes. It is used to mark variables as input for the module. 
     */
    virtual void addInput(std::shared_ptr<Variable> input, std::uint32_t units){};
    /**
     * @brief virtual function that is supposed to be implemented by the derived classes. It is used to mark variables as output for the module.
     */
    virtual void addOutput(std::shared_ptr<Variable> output){};
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
    static void setGraph(std::shared_ptr<Graph> graph){GRAPH = graph;}
    /**
     * @brief used to get the private variable mUnits, could be moved to the respective derived classes.
     */
    std::uint32_t getUnits()
    {
        if(mUnits == -1)throw std::runtime_error("units not set");
        return mUnits;
    }
};


// this is mainly for interface purposes


// code of all child classes
#include "input.hpp"
#include "dense.hpp"
#include "cost.hpp"
#include "output.hpp"

using ModuleVariant = std::variant<Dense>;

#endif // MODULE_HPP