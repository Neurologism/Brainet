#ifndef NORM_INCLUDE_GUARD
#define NORM_INCLUDE_GUARD

#include "./module.h"

/**
 * @brief Used to add a norm penalty to the graph. This is used to prevent overfitting.
 */
class NORM : public MODULE
{
    std::shared_ptr<VARIABLE> _norm_function;
    std::shared_ptr<VARIABLE> _total_cost;

public:

    /**
     * @brief add a norm penalty to the graph
     * @param norm the norm to be used
     * @param lambda the lambda value to be used
     */
    NORM(NORM_VARIANT norm, double lambda);
    ~NORM() = default;
    /**
     * @brief used to mark variables as input for the module.
     */
    void add_input(std::shared_ptr<VARIABLE> input, std::uint32_t input_units) override
    {
        _norm_function->get_inputs().push_back(input);
    }
    /**
     * @brief used to mark variables as output for the module.
     */
    void add_output(std::shared_ptr<VARIABLE> output) override
    {
        _norm_function->get_consumers().push_back(output);
    }
    /**
     * @brief used to get the input variables of the module specified by the index.
     */
    std::shared_ptr<VARIABLE> input(std::uint32_t index) override
    {
        return _norm_function;
    }
    /**
     * @brief used to get the output variables of the module specified by the index.
     */
    std::shared_ptr<VARIABLE> output(std::uint32_t index) override
    {
        return _norm_function;
    }
};

NORM::NORM(NORM_VARIANT norm, double lambda)
{
    _norm_function = std::make_shared<VARIABLE>(VARIABLE(/*operation computing the norm*/,{},{}));
    _total_cost = std::make_shared<VARIABLE>(VARIABLE(/*operation computing the final cost*/,{_norm_function},{}));

    _total_cost->get_inputs().push_back(_norm_function);
    __learnable_parameters.push_back(_norm_function);
    __learnable_parameters.push_back(_total_cost);
}

#endif // NORM_INCLUDE_GUARD