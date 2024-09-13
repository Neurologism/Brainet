#ifndef MODULE_HPP
#define MODULE_HPP

#include "input.hpp"
#include "../dependencies.hpp"
#include "../variable.hpp"
#include "fullyconnected/output.hpp"

/**
 * @brief The Module class can be used to group multiple variables together. This is useful for creating structures of variables with associated operations. 
 */
class Module
{
protected:
    std::uint32_t mUnits = -1; // stores the input size of the module
    // (should be set in the constructor) moves to child classes?
    std::string mName; // stores the name of the module used for simple identification

public:
    explicit Module(std::string  name) : mName(std::move(name)) {}
    virtual ~Module() = default;

    /**
     * @brief add an input to the module
     * @param input the input variable
     * @param inputSize the size of the input
     */
    virtual void addInput(const std::shared_ptr<Variable> &input, const std::uint32_t &inputSize) = 0;

    /**
     * @brief add an output to the module
     * @param output the output variable
     */
    virtual void addOutput(const std::shared_ptr<Variable>& output) = 0;


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

using ModuleVariant = std::variant<Input, Dense, Output>;
using HiddenModuleVariant = std::variant<Dense>;

#endif // MODULE_HPP