#ifndef INPUT_INCLUDE_GUARD
#define INPUT_INCLUDE_GUARD

#include ".\cluster.h"

/**
 * @brief this can store the input data of the model. Initalize with a pointer to the data and update the data when needed. This owns only 1 variable and does nothing else.
 * Preprocessing could be added at this point in the future.
 */
class INPUT : public CLUSTER
{
    std::shared_ptr<VARIABLE> _input_variable;
public:
    /**
     * @brief add an input to the graph
     * @param data a pointer to the input data
     * @param units the respective size of a single input
     */
    INPUT(std::shared_ptr<TENSOR<double>> & data, int units);
    ~INPUT() = default;
    /**
     * @brief throw an error if this function is called because the input variable cannot have an input.
     */
    void add_input(std::shared_ptr<VARIABLE> input, int units) override;
    /**
     * @brief used to mark variables as output for the cluster.
     */
    void add_output(std::shared_ptr<VARIABLE> output) override;
    /**
     * @brief throw an error if this function is called because the input variable cannot have an input.
     */
    std::shared_ptr<VARIABLE> input(int index) override;
    /**
     * @brief used to get the output variables of the cluster specified by the index.
     */
    std::shared_ptr<VARIABLE> output(int index) override;
};

INPUT::INPUT(std::shared_ptr<TENSOR<double>> & data, int units)
{
    // error checks
    if(__graph == nullptr)
    {
        throw std::runtime_error("graph is not set");
    }
    
    // create the input variable
    _input_variable = __graph->add_variable(std::make_shared<VARIABLE>(VARIABLE(nullptr, {}, {}, data)));
    __units = units; // set the number of neurons in the layer
}

void INPUT::add_input(std::shared_ptr<VARIABLE> input, int units)
{
    throw std::runtime_error("Input variable cannot have an input");
}

void INPUT::add_output(std::shared_ptr<VARIABLE> output)
{
    _input_variable->get_consumers().push_back(output);
}

std::shared_ptr<VARIABLE> INPUT::input(int index)
{
    throw std::runtime_error("Input variable cannot have an input");
    return nullptr;
}

std::shared_ptr<VARIABLE> INPUT::output(int index)
{
    return _input_variable;
}
#endif // INPUT_INCLUDE_GUARD