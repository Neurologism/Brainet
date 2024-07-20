#ifndef COST_INCLUDE_GUARD
#define COST_INCLUDE_GUARD

#include "dependencies.h"
#include "cluster.h"
#include "../operation/cost_function/cost_function.h"

class COST : public CLUSTER
{
    std::shared_ptr<VARIABLE> _output_variable;

public:
    COST(COST_FUNCTION_VARIANT cost_function);
    ~COST() = default;
    void add_input(std::shared_ptr<VARIABLE> input, int units) override
    {
        _output_variable->get_inputs().push_back(input);
    }
    void add_output(std::shared_ptr<VARIABLE> output) override
    {
        _output_variable->get_consumers().push_back(output);
    }
    std::shared_ptr<VARIABLE> input(int index) override
    {
        return _output_variable;
    }
    std::shared_ptr<VARIABLE> output(int index) override
    {
        return _output_variable;
    }
};

COST::COST(COST_FUNCTION_VARIANT cost_function)
{
    if(__graph == nullptr)
    {
        throw std::runtime_error("graph is not set");
    }

    // create the variables
    _output_variable = __graph->add_variable(std::make_shared<VARIABLE>(VARIABLE(std::visit([](auto&& arg) {
        return std::shared_ptr<OPERATION>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, COST_FUNCTION_VARIANT{cost_function}), {}, {})));
}



    

#endif // COST_INCLUDE_GUARD