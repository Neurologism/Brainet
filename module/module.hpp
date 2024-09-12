#ifndef MODULE_HPP
#define MODULE_HPP

#include <utility>

#include "../dependencies.hpp"
#include "../variable.hpp"

/**
 * @brief The Module class can be used to group multiple variables together. This is useful for creating structures of variables with associated operations. 
 */
class Module
{
protected:
    std::uint32_t mUnits = -1; // stores the input size of the module (should be set in the constructor) move to child classes?
    std::string mName; // stores the name of the module used for simple identification

public:
    explicit Module(std::string  name) : mName(std::move(name)) {}
    virtual ~Module() = default;

    /**
     * @brief used to initialize the module. This is used to add initial connections to other modules.
     * @param initialInputs the input variables
     * @param initialOutputs the output variables
     */
    virtual void __init__( std::vector<std::shared_ptr<Variable>> initialInputs, std::vector<std::shared_ptr<Variable>> initialOutputs ) = 0;

    /**
     * @brief function to get access to specific variables of the module.
     * @param index the index of the variable
     * @return the variable specified by the index
     */
    virtual std::shared_ptr<Variable> getVariable(std::uint32_t index) = 0;

    /**
     * @brief used to get the private variable mUnits
     */
    [[nodiscard]] std::uint32_t getUnits() const
    {
        if(mUnits == -1)throw std::runtime_error("units not set");
        return mUnits;
    }

    /**
     * @brief used to get the name of the module
     */
    std::string getName()
    {
        return mName;
    }
};


// this is mainly for interface purposes


// code of all child classes
#include "fullyconnected/dense.hpp"

using ModuleVariant = std::variant<Dense>;

#endif // MODULE_HPP