#ifndef COST_INCLUDE_GUARD
#define COST_INCLUDE_GUARD

#include "../dependencies.h"
#include "module.h"
#include "../operation/cost_function/cost_function.h"
#include "../operation/processing/one_hot.h"

/**
 * @brief the cost module is intended for calculating the cost of various models. Currently only cost functions dependy on the output of the network and the y truth are supported.
 
 */
class COST : public MODULE
{
    std::shared_ptr<VARIABLE> _target_variable; // storing y truth
    std::shared_ptr<VARIABLE> _output_variable; // actual cost funtion 
    std::shared_ptr<VARIABLE> _one_hot_variable; // one hot encoding of the y truth

public:
    /**
     * @brief add a cost function to the graph
     * @param cost_function the operation representing the cost function.
     * @param one_hot_ecoding_size the size of the one hot encoding. Default is 0. One hot encoding assumes that the y truth is initially a single value giving the index of the on value.
     * @param one_hot_ecoding_on_value the value to set the one hot encoding to. Default is 1.
     * @param one_hot_ecoding_off_value the value to set the rest of the tensor to. Default is 0.
     */
    COST(COST_FUNCTION_VARIANT cost_function, std::uint32_t one_hot_ecoding_size = 0, std::uint32_t one_hot_ecoding_on_value = 1, std::uint32_t one_hot_ecoding_off_value = 0);
    /**
     * @brief add a cost function to the graph
     * @param cost_function the operation representing the cost function.
     * @param one_hot_ecoding_size the size of the one hot encoding. Default is 0. One hot encoding assumes that the y truth is initially a single value giving the index of the on value.
     * @param label_smoothing the value to smooth the labels. Default is 0.
     */
    COST(COST_FUNCTION_VARIANT cost_function, std::uint32_t one_hot_ecoding_size, double label_smoothing);
    ~COST() = default;

    /**
     * @brief used to mark variables as input for the module.
     */
    void add_input(std::shared_ptr<VARIABLE> input, std::uint32_t units) override
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
    std::shared_ptr<VARIABLE> input(std::uint32_t index) override
    {
        return _output_variable;
    }
    /**
     * @brief used to get the output variables of the module specified by the index.
     */
    std::shared_ptr<VARIABLE> output(std::uint32_t index) override
    {
        return _output_variable;
    }
    /**
     * @brief used to get the target variable of the module.
     */
    std::shared_ptr<VARIABLE> target()
    {
        return _target_variable;
    }
};

COST::COST(COST_FUNCTION_VARIANT cost_function, std::uint32_t one_hot_ecoding_size, std::uint32_t one_hot_ecoding_on_value, std::uint32_t one_hot_ecoding_off_value)
{
    // error checks
    if(__graph == nullptr)
    {
        throw std::runtime_error("graph is not set");
    }


    // add variables to the graph
    _target_variable = __graph->add_variable(std::make_shared<VARIABLE>(VARIABLE(nullptr, {}, {})));

    if (one_hot_ecoding_size == 0) // check if 1 hot encoding requested
    {
        _one_hot_variable = nullptr;
        
        // add variables to the graph
        _target_variable = __graph->add_variable(std::make_shared<VARIABLE>(VARIABLE(nullptr, {}, {})));

        _output_variable = __graph->add_variable(std::make_shared<VARIABLE>(VARIABLE(std::visit([](auto&& arg) {
            return std::shared_ptr<OPERATION>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, COST_FUNCTION_VARIANT{cost_function}), {_target_variable}, {})));
        
        // connections within the module
        _target_variable->get_consumers().push_back(_output_variable);
    }
    else
    {
        // variable performing one hot encoding; no backward pass is supported
        _one_hot_variable = __graph->add_variable(std::make_shared<VARIABLE>(VARIABLE(std::make_shared<OneHot>(OneHot(10, 1, 0)), {_target_variable}, {}))); 

        // conversion of the COST_FUNCTION_VARIANT to an operation pointer
        _output_variable = __graph->add_variable(std::make_shared<VARIABLE>(VARIABLE(std::visit([](auto&& arg) {
            return std::shared_ptr<OPERATION>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, COST_FUNCTION_VARIANT{cost_function}), {_one_hot_variable}, {})));
        
        // connections within the module
        _target_variable->get_consumers().push_back(_one_hot_variable);
        _one_hot_variable->get_consumers().push_back(_output_variable);
    }
}

COST::COST(COST_FUNCTION_VARIANT cost_function, std::uint32_t one_hot_ecoding_size, double label_smoothing)
{
    if (label_smoothing <= 0)
    {
        throw std::invalid_argument("COST::COST: label_smoothing must be greater than 0");
    }
    if (label_smoothing >= 1)
    {
        throw std::invalid_argument("COST::COST: label_smoothing must be less than 1");
    }
    COST(cost_function, one_hot_ecoding_size, 1-label_smoothing, label_smoothing/(one_hot_ecoding_size-1));
}
#endif // COST_INCLUDE_GUARD