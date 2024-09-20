#ifndef PARAMETER_NORM_PENALTY_HPP
#define PARAMETER_NORM_PENALTY_HPP

#include "../operation.hpp"


/**
 * @brief Base class for operation functions. Template class to create a parameternormpenalty penalty that executes elementwise.
 */
class ParameterNormPenalty : public Operation
{
protected:
    double _lambda;

public:
    /**
     * @brief add a parameter norm penalty to the graph
     * @param lambda the lambda value to be used
     */
    ParameterNormPenalty(double lambda) : _lambda(lambda) {}
    ~ParameterNormPenalty() = default;

    virtual void f(std::vector<std::shared_ptr<Variable>>& inputs) = 0;
    virtual std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient) = 0;
};

#endif // PARAMETER_NORM_PENALTY_HPP