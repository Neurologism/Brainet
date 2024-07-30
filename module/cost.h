#ifndef COST_INCLUDE_GUARD
#define COST_INCLUDE_GUARD

#include "../dependencies.h"
#include "module.h"
#include "../operation/cost_function/cost_function.h"

/**
 * @brief the cost module is intended for calculating the cost of various models. Currently only cost functions dependy on the output of the network and the y truth are supported.
 
 */
class COST : public MODULE
{
    std::shared_ptr<VARIABLE> _target_variable; // storing y truth
    std::shared_ptr<VARIABLE> _output_variable; // actual cost funtion 

public:
    /**
     * @brief add a cost function to the graph
     * @param cost_function the operation representing the cost function.
     * @param data a pointer to the y truth of the model (the target variable) If the y truth should change please change the satelite data.
     */
    COST(COST_FUNCTION_VARIANT cost_function);
    ~COST() = default;

    /**
     * @brief used to mark variables as input for the module.
     */
    void add_input(std::shared_ptr<VARIABLE> input, int units) override
    {
        _output_variable->get_inputs().push_back(input);
    }
    /**
     * @brief used to mark variables as output for the module.
     */
    void add_output(std::shared_ptr<VARIABLE> output) override
    {
        _output_variable->get_consumers().push_back(output);
    }
    /**
     * @brief used to get the input variables of the module specified by the index.
     */
    std::shared_ptr<VARIABLE> input(int index) override
    {
        return _output_variable;
    }
    /**
     * @brief used to get the output variables of the module specified by the index.
     */
    std::shared_ptr<VARIABLE> output(int index) override
    {
        return _output_variable;
    }
};

COST::COST(COST_FUNCTION_VARIANT cost_function)
{
    // error checks
    if(__graph == nullptr)
    {
        throw std::runtime_error("graph is not set");
    }


    // add variables to the graph
    _target_variable = __graph->add_variable(std::make_shared<VARIABLE>(VARIABLE(nullptr, {}, {})));

    _output_variable = __graph->add_variable(std::make_shared<VARIABLE>(VARIABLE(std::visit([](auto&& arg) {
        return std::shared_ptr<OPERATION>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, COST_FUNCTION_VARIANT{cost_function}), {_target_variable}, {})));
    
    // connections within the module
    _target_variable->get_consumers().push_back(_output_variable);
}



    

#endif // COST_INCLUDE_GUARD