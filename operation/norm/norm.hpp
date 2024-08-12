#ifndef NORM_HPP
#define NORM_HPP

#include "../operation.hpp"


/**
 * @brief Base class for operation functions. Template class to create a norm penalty that executes elementwise.
 */
class Norm : public Operation
{
protected:
    double _lambda;

public:
    /**
     * @brief add a norm penalty to the graph
     * @param lambda the lambda value to be used
     */
    Norm(double lambda) : _lambda(lambda) {}
    ~Norm() = default;

    virtual void f(std::vector<std::shared_ptr<Variable>>& inputs) = 0;
    virtual std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient) = 0;
};


#include "L1.hpp"
#include "L2.hpp"

using NormVariant = std::variant<L1, L2>;


#endif // NORM_HPP