#ifndef OPERATION_INCLUDE_GUARD
#define OPERATION_INCLUDE_GUARD

#include "../dependencies.h"
#include "../variable.h"
#include "../tensor.h"

class VARIABLE;

/**
 * @brief OPERATION class is a wrapper class for the mathematical operations that are performed on the variables.
*/
class OPERATION
{
private:
    std::shared_ptr<VARIABLE> __variable = nullptr; // necessary for storing the result of the operation
public:
    OPERATION() = default;
    virtual ~OPERATION() = default;
    /**
     * @brief mathematical function the operation implements
    */
    virtual void f(std::vector<std::shared_ptr<VARIABLE>>& inputs) =0; 
    /**
     * @brief derivative of the function
     * assumes that the gradient is already calculated for the output variables 
     * @param inputs the parents of the variable
     * @param focus this is the only variable everything else is constant
     * @param gradient the sum of the gradients of the consumers
    */
    virtual std::shared_ptr<TENSOR<double>> bprop(std::vector<std::shared_ptr<VARIABLE>>& inputs, std::shared_ptr<VARIABLE> & focus, std::shared_ptr<TENSOR<double>> & gradient) =0;
    /**
     * @brief sets the variable of the operation
    */
    void set_variable(std::shared_ptr<VARIABLE> variable);
    /**
     * @brief returns the variable of the operation
    */
    std::shared_ptr<VARIABLE> get_variable();
};

void OPERATION::set_variable(std::shared_ptr<VARIABLE> variable)
{
    __variable = variable;
}

std::shared_ptr<VARIABLE> OPERATION::get_variable()
{
    if(__variable == nullptr)
    {
        throw std::runtime_error("variable is not set"); // this should never happen
    }
    return __variable;
}

#endif // OPERATION_INCLUDE_GUARD