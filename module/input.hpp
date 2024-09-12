#ifndef INPUT_HPP
#define INPUT_HPP

#include <graph.hpp>

#include "./module.hpp"
#include "../operation/processing/noise.hpp"
#include "../operation/processing/dropout.hpp"

/**
 * @brief this can store the input data of the model. Initialize with a pointer to the data and update the data when needed. This owns only 1 variable and does nothing else.
 * Preprocessing could be added at this point in the future.
 */
class Input final : public Module
{
    std::shared_ptr<Variable> mInputVariable;
    std::shared_ptr<Variable> mNoiseVariable;
    std::shared_ptr<Variable> mDropoutVariable;

public:
    /**
     * @brief add an input to the graph
     * @param units the respective size of a single input
     * @param name the name of the module
     * @param dropout the dropout rate of the input
     */
    explicit Input(std::uint32_t units, const std::string &name = "", double dropout = 1.0);
    
    /**
     * @brief add an input to the graph with a noise operation
     * @param units the respective size of a single input
     * @param noise the noise operation to add to the input
     * @param dropout the dropout rate of the input
     */
    Input(std::uint32_t units, const Noise& noise, double dropout = 1.0);

    ~Input() override = default;

    /**
     * @brief used to initialize the module with the input and output variables.
     * @param initialInputs the initial input variables
     * @param initialOutputs the initial output variables
     */
    void __init__( std::vector<std::shared_ptr<Variable>> initialInputs, std::vector<std::shared_ptr<Variable>> initialOutputs ) override;

    /**
     * @brief gives access to all variables of the module.
     * @param index the index of the variable
     * @return the variable specified by the index
     * @note 0: InputVariable
     * @note 1: DropoutVariable
     */
    std::shared_ptr<Variable> getVariable(std::uint32_t index) override;    
};

inline Input::Input(const std::uint32_t units, const std::string &name, const double dropout) : Module(name)
{
    // create the input variable
    mInputVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {})));
    mUnits = units; // set the number of neurons in the layer

    // create the dropout variable
    mDropoutVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Dropout>(Dropout(dropout)), {})));
}

inline Input::Input(std::uint32_t units, const Noise& noise, double dropout) : Input(units, "", dropout)
{
    // create the noise variable
    mNoiseVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Noise>(noise), {mInputVariable}, {mDropoutVariable})));
}


void Input::__init__(const std::vector<std::shared_ptr<Variable>> initialInputs, const std::vector<std::shared_ptr<Variable>> initialOutputs)
{
    if (!initialInputs.empty())
    {
        throw std::runtime_error("Input module cannot have inputs");
    }

    if (initialOutputs.size() != 1)
    {
        throw std::runtime_error("Input module must have exactly one output");
    }

    if(mNoiseVariable != nullptr)
    {
        mInputVariable->getConsumers().push_back(mNoiseVariable);
        mDropoutVariable->getInputs().push_back(mNoiseVariable);
    }
    else
    {
        mInputVariable->getConsumers().push_back(mDropoutVariable);
        mDropoutVariable->getInputs().push_back(mInputVariable);
    }   
    mDropoutVariable->getConsumers().push_back(initialOutputs[0]);
}

inline std::shared_ptr<Variable> Input::getVariable(const std::uint32_t index)
{
    switch (index)
    {
    case 0:
        return mInputVariable;
    case 1:
        return mDropoutVariable;
    default:
        throw std::runtime_error("index out of bounds");
    }
}

#endif // INPUT_HPP