#ifndef OPERATION_HPP
#define OPERATION_HPP

#include "variable.hpp"
#include "../datatypes/tensor.hpp"
#include "../datatypes/matrix.hpp"
#include "../datatypes/vector.hpp"

class Variable;

/**
 * @brief Operation class is a wrapper class for the mathematical operations that are performed to calculate the tensor value of a variable.
 */
class Operation
{
    std::shared_ptr<Variable> mpVariable = nullptr; // necessary for storing the result of the operation

protected:
    std::string mName = "Operation"; // name of the operation

public:
    Operation() = default;
    ~Operation() = default;

    /**
     * @brief mathematical function the operation implements
     */
    virtual void f(std::vector<std::shared_ptr<Variable>> &inputs) = 0;

    /**
     * @brief the derivative of the function
     * assumes that the gradient is already calculated for the output variables
     * @param inputs the parents of the variable
     * @param focus this is the only variable everything else is constant
     * @param gradient the sum of the consumers gradients
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

#endif // OPERATION_HPP