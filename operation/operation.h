#ifndef OPERATION_INCLUDE_GUARD
#define OPERATION_INCLUDE_GUARD

#include "..\dependencies.h"
#include "..\variable.h"
#include "..\tensor.h"

class VARIABLE;

/**
 * @brief OPERATION class is an abstract class that defines the interface for all operations.
*/
class OPERATION
{
private:
    std::shared_ptr<VARIABLE> __variable = nullptr;
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
     * @param inputs the input variables of the operation
     * @param focus the variable that the gradient is calculated for
     * @param gradient the sum of the gradients of the output variables
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
        throw std::runtime_error("variable is not set");
    }
    return __variable;
}

#endif // OPERATION_INCLUDE_GUARD