#ifndef INPUT_INCLUDE_GUARD
#define INPUT_INCLUDE_GUARD

#include ".\cluster.h"

/**
 * @brief INPUT class creates an input variable for the graph
 * it is a wrapper class for a variable that is not the result of an operation
 */
class INPUT : public CLUSTER
{
    std::shared_ptr<VARIABLE> _input_variable;
public:
    INPUT(TENSOR<double> & data, int units);
    ~INPUT() = default;
    void add_input(std::shared_ptr<VARIABLE> input, int units) override;
    void add_output(std::shared_ptr<VARIABLE> output) override;
    std::shared_ptr<VARIABLE> input(int index) override;
    std::shared_ptr<VARIABLE> output(int index) override;
};

INPUT::INPUT(TENSOR<double> & data, int units)
{
    if(__graph == nullptr)
    {
        throw std::runtime_error("graph is not set");
    }
    
    _input_variable = __graph->add_variable(std::make_shared<VARIABLE>(VARIABLE(nullptr, {}, {}, std::make_shared<TENSOR<double>>(data))));
    __units = units;
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