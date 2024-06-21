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
    VARIABLE * __variable = nullptr;
public:
    OPERATION() = default;
    /**
     * @brief mathematical function the operation implements
    */
    virtual void f(std::vector<VARIABLE *>& inputs) =0; 
    /**
     * @brief derivative of the function
     * assumes that the gradient is already calculated for the output variables 
     * @param inputs the input variables of the operation
     * @param focus the variable that the gradient is calculated for
     * @param gradient the sum of the gradients of the output variables
    */
    virtual TENSOR<double> bprop(std::vector<VARIABLE *>& inputs, VARIABLE & focus, TENSOR<double> & gradient) =0;
    /**
     * @brief sets the variable of the operation
    */
    void set_variable(VARIABLE * variable);
    /**
     * @brief returns the variable of the operation
    */
    VARIABLE * get_variable(){return __variable;};
};

void OPERATION::set_variable(VARIABLE * variable)
{
    __variable = variable;
}

VARIABLE * OPERATION::get_variable()
{
    if(__variable == nullptr)
    {
        throw std::runtime_error("variable is not set");
    }
    return __variable;
}

#endif // OPERATION_INCLUDE_GUARD