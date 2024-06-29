#ifndef INPUT_INCLUDE_GUARD
#define INPUT_INCLUDE_GUARD

#include ".\cluster.h"

/**
 * @brief INPUT class creates an input variable for the graph
 * it is a wrapper class for a variable that is not the result of an operation
 */
class INPUT : public CLUSTER
{
    VARIABLE * _input_variable;
public:
    INPUT(TENSOR<double> & data, int units);
    void add_input(VARIABLE * input, int units) override;
    void add_output(VARIABLE * output) override;
    VARIABLE * input(int index) override;
    VARIABLE * output(int index) override;
};

INPUT::INPUT(TENSOR<double> & data, int units)
{
    if(__graph == nullptr)
    {
        throw std::runtime_error("graph is not set");
    }
    _input_variable = __graph->add_variable(VARIABLE(nullptr, {}, {}, data));
}

void INPUT::add_input(VARIABLE * input, int units)
{
    throw std::runtime_error("Input variable cannot have an input");
}

void INPUT::add_output(VARIABLE * output)
{
    _input_variable->get_consumers()->push_back(output);
}

VARIABLE * INPUT::input(int index)
{
    throw std::runtime_error("Input variable cannot have an input");
    return nullptr;
}

VARIABLE * INPUT::output(int index)
{
    return _input_variable;
}
#endif // INPUT_INCLUDE_GUARD