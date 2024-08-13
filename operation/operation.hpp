#ifndef OPERATION_HPP
#define OPERATION_HPP

#include "../variable.hpp"
#include "../datatypes/tensor.hpp"
#include "../datatypes/matrix.hpp"
#include "../datatypes/vector.hpp"

class Variable;

/**
 * @brief Operation class is a wrapper class for the mathematical operations that are performed on the variables.
 */
class Operation
{
    std::shared_ptr<Variable> mpVariable = nullptr; // necessary for storing the result of the operation

protected:
    std::string mName = "Operation"; // name of the operation

public:
    Operation() = default;
    virtual ~Operation() = default;

    /**
     * @brief mathematical function the operation implements
     */
    virtual void f(std::vector<std::shared_ptr<Variable>> &inputs) = 0;

    /**
     * @brief derivative of the function
     * assumes that the gradient is already calculated for the output variables
     * @param inputs the parents of the variable
     * @param focus this is the only variable everything else is constant
     * @param gradient the sum of the gradients of the consumers
     */
    virtual std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>> &inputs, std::shared_ptr<Variable> &focus, std::shared_ptr<Tensor<double>> &gradient) = 0;

    /**
     * @brief sets the variable of the operation
     */
    void setVariable(std::shared_ptr<Variable> variable);

    /**
     * @brief returns the variable of the operation
     */
    std::shared_ptr<Variable> getVariable();

    /**
     * @brief returns the name of the operation
     */
    std::string getName()
    {
        return mName;
    }
};

void Operation::setVariable(std::shared_ptr<Variable> variable)
{
    mpVariable = variable;
}

std::shared_ptr<Variable> Operation::getVariable()
{
    if (mpVariable == nullptr)
    {
        throw std::runtime_error("variable is not set"); // this should never happen
    }
    return mpVariable;
}

#endif // OPERATION_HPP