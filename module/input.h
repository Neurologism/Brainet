#ifndef INPUT_INCLUDE_GUARD
#define INPUT_INCLUDE_GUARD

#include "./module.h"
#include "../operation/processing/noise.h"

/**
 * @brief this can store the input data of the model. Initalize with a pointer to the data and update the data when needed. This owns only 1 variable and does nothing else.
 * Preprocessing could be added at this point in the future.
 */
class INPUT : public MODULE
{
    std::shared_ptr<Variable> _input_variable;
    std::shared_ptr<Variable> _noise_variable;
public:
    /**
     * @brief add an input to the graph
     * @param units the respective size of a single input
     */
    INPUT(std::uint32_t units);
    /**
     * @brief add an input to the graph with a noise operation
     * @param units the respective size of a single input
     * @param noise the noise operation to add to the input
     */
    INPUT(std::uint32_t units, Noise noise);
    ~INPUT() = default;
    /**
     * @brief throw an error if this function is called because the input variable cannot have an input.
     */
    void add_input(std::shared_ptr<Variable> input, std::uint32_t units) override;
    /**
     * @brief used to mark variables as output for the module.
     */
    void add_output(std::shared_ptr<Variable> output) override;
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

INPUT::INPUT(std::uint32_t units)
{
    // error checks
    if(__graph == nullptr)
    {
        throw std::runtime_error("graph is not set");
    }
    
    // create the input variable
    _input_variable = __graph->add_variable(std::make_shared<Variable>(Variable(nullptr, {}, {})));
    __units = units; // set the number of neurons in the layer

    _noise_variable = nullptr; // no noise is added
}

INPUT::INPUT(std::uint32_t units, Noise noise)
{
    // error checks
    if(__graph == nullptr)
    {
        throw std::runtime_error("graph is not set");
    }
    
    // create the input variable
    _input_variable = __graph->add_variable(std::make_shared<Variable>(Variable(nullptr, {}, {})));
    __units = units; // set the number of neurons in the layer

    // create the noise variable
    _noise_variable = __graph->add_variable(std::make_shared<Variable>(Variable(std::make_shared<Noise>(noise), {_input_variable})));
    
    _input_variable->get_consumers().push_back(_noise_variable); // add the noise variable as a consumer of the input variable
}

void INPUT::add_input(std::shared_ptr<Variable> input, std::uint32_t units)
{
    throw std::runtime_error("Input variable cannot have an input");
}

void INPUT::add_output(std::shared_ptr<Variable> output)
{
    if(_noise_variable != nullptr)
    {
        _noise_variable->get_consumers().push_back(output);
    }
    else
    {
        _input_variable->get_consumers().push_back(output);
    }
}

std::shared_ptr<Variable> INPUT::input(std::uint32_t index)
{
    throw std::runtime_error("Input variable cannot have an input");
    return nullptr;
}

std::shared_ptr<Variable> INPUT::output(std::uint32_t index)
{
    if(_noise_variable != nullptr)
    {
        return _noise_variable;
    }
    return _input_variable;
}

std::shared_ptr<Variable> INPUT::data()
{
    return _input_variable;
}

#endif // INPUT_INCLUDE_GUARD