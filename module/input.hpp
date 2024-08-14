#ifndef INPUT_HPP
#define INPUT_HPP

#include "./module.hpp"
#include "../operation/processing/noise.hpp"

/**
 * @brief this can store the input data of the model. Initalize with a pointer to the data and update the data when needed. This owns only 1 variable and does nothing else.
 * Preprocessing could be added at this point in the future.
 */
class Input : public Module
{
    std::shared_ptr<Variable> mInputVariable;
    std::shared_ptr<Variable> mNoiseVariable;
public:
    /**
     * @brief add an input to the graph
     * @param units the respective size of a single input
     */
    Input(std::uint32_t units);
    /**
     * @brief add an input to the graph with a noise operation
     * @param units the respective size of a single input
     * @param noise the noise operation to add to the input
     */
    Input(std::uint32_t units, Noise noise);
    ~Input() = default;
    /**
     * @brief throw an error if this function is called because the input variable cannot have an input.
     */
    void addInput(std::shared_ptr<Variable> input, std::uint32_t units) override;
    /**
     * @brief used to mark variables as output for the module.
     */
    void addOutput(std::shared_ptr<Variable> output) override;
    /**
     * @brief throw an error if this function is called because the input variable cannot have an input.
     */
    std::shared_ptr<Variable> input(std::uint32_t index) override;
    /**
     * @brief used to get the output variables of the module specified by the index.
     */
    std::shared_ptr<Variable> output(std::uint32_t index) override;
    /**
     * @brief used to get the variable used to load the data.
     */
    std::shared_ptr<Variable> data();
    
};

Input::Input(std::uint32_t units)
{
    // error checks
    if(GRAPH == nullptr)
    {
        throw std::runtime_error("graph is not set");
    }
    
    // create the input variable
    mInputVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {})));
    mUnits = units; // set the number of neurons in the layer

    mNoiseVariable = nullptr; // no noise is added
}

Input::Input(std::uint32_t units, Noise noise)
{
    // error checks
    if(GRAPH == nullptr)
    {
        throw std::runtime_error("graph is not set");
    }
    
    // create the input variable
    mInputVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {})));
    mUnits = units; // set the number of neurons in the layer

    // create the noise variable
    mNoiseVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::make_shared<Noise>(noise), {mInputVariable})));
    
    mInputVariable->getConsumers().push_back(mNoiseVariable); // add the noise variable as a consumer of the input variable
}

void Input::addInput(std::shared_ptr<Variable> input, std::uint32_t units)
{
    throw std::runtime_error("Input variable cannot have an input");
}

void Input::addOutput(std::shared_ptr<Variable> output)
{
    if(mNoiseVariable != nullptr)
    {
        mNoiseVariable->getConsumers().push_back(output);
    }
    else
    {
        mInputVariable->getConsumers().push_back(output);
    }
}

std::shared_ptr<Variable> Input::input(std::uint32_t index)
{
    throw std::runtime_error("Input variable cannot have an input");
    return nullptr;
}

std::shared_ptr<Variable> Input::output(std::uint32_t index)
{
    if(mNoiseVariable != nullptr)
    {
        return mNoiseVariable;
    }
    return mInputVariable;
}

std::shared_ptr<Variable> Input::data()
{
    return mInputVariable;
}

#endif // INPUT_HPP