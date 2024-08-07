#ifndef NORM_INCLUDE_GUARD
#define NORM_INCLUDE_GUARD

#include "../operation.h"


/**
 * @brief Base class for operation functions. Template class to create a norm penalty that executes elementwise.
 */
class NORM : public OPERATION
{
protected:
    double _lambda;

public:
    /**
     * @brief add a norm penalty to the graph, using a default lambda value
     */
    NORM() : _lambda(_default_lambda) {};
    /**
     * @brief add a norm penalty to the graph
     * @param lambda the lambda value to be used
     */
    NORM(double lambda) : _lambda(lambda) {}
    ~NORM() = default;

    virtual void f(std::vector<std::shared_ptr<VARIABLE>>& inputs) = 0;
    virtual std::shared_ptr<TENSOR<double>> bprop(std::vector<std::shared_ptr<VARIABLE>>& inputs, std::shared_ptr<VARIABLE> & focus, std::shared_ptr<TENSOR<double>> & gradient) = 0;
};


#include "L1.h"
#include "L2.h"

using NORM_VARIANT = std::variant<L1_NORM, L2_NORM>;


#endif // NORM_INCLUDE_GUARD