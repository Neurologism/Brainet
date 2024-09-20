#ifndef MODULE_HPP
#define MODULE_HPP

#include "graph.hpp"

/**
 * @brief The Module class can be used to group multiple variables together. This is useful for creating structures of variables with associated operations. 
 */
class Module
{
    std::string mName; // stores the name of the module used for simple identification

public:
    explicit Module(std::string  name) : mName(std::move(name)) {}
    virtual ~Module() = default;

    virtual std::vector<std::shared_ptr<Variable>> getInputs() = 0;
    virtual std::vector<std::shared_ptr<Variable>> getOutputs() = 0;
    virtual std::vector<std::shared_ptr<Variable>> getLearnableVariables() = 0;
    virtual std::vector<std::shared_ptr<Variable>> getGradientVariables() = 0;

    /**
     * @brief used to get the name of the module
     */
    std::string getName();
};

// code of all child classes
#include "dataset.hpp"
#include "layer.hpp"
#include "loss.hpp"

using ModuleVariant = std::variant<Dense, Loss, Dataset>;

#endif // MODULE_HPP