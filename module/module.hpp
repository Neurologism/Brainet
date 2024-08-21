#ifndef MODULE_HPP
#define MODULE_HPP

#include "../dependencies.hpp"
#include "../variable.hpp"
#include "../graph.hpp"

/**
 * @brief The Module class can be used to group multiple variables together. This is useful for creating structures of variables with associated operations. 
 */
class Module
{
protected:
    std::uint32_t mUnits = -1; // stores the input size of the module (should be set in the constructor) move to child classes?
public:
    virtual ~Module() = default;

    /**
     * @brief used to initialize the module. This is used to add initial connections to other modules.
     * @param initialInpus the input variables
     * @param initialOutputs the output variables
     */
    virtual void __init__( std::vector<std::shared_ptr<Variable>> initialInpus, std::vector<std::shared_ptr<Variable>> initialOutputs ) = 0;

    /**
     * @brief function to get access to specific variables of the module.
     * @param index the index of the variable
     * @return the variable specified by the index
     */
    virtual std::shared_ptr<Variable> getVariable(std::uint32_t index) = 0;

    /**
     * @brief used to get the private variable mUnits
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
#include "fullyconnected/dense.hpp"
#include "loss.hpp"
#include "fullyconnected/output.hpp"
#include "ensemble_module.hpp"

using ModuleVariant = std::variant<Dense>;

#endif // MODULE_HPP