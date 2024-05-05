#ifndef OPERATION_INCLUDE_GUARD
#define OPERATION_INCLUDE_GUARD

#include "..\dependencies.h"
#include "..\abstract_variable.h"

/**
 * @brief OPERATION class is an abstract class that defines the interface for all operations.
*/
class OPERATION
{
public:
    OPERATION() = default;
    /**
     * @brief mathematical function the operation implements
    */
    virtual void f(std::vector<ABS_VARIABLE *>& inputs) =0; 
    /**
     * @brief derivative of the function
     * assumes that the gradient is already calculated for the output variables 
     * @param inputs the input variables of the operation
     * @param focus the variable that the gradient is calculated for
     * @param outputs the output variables of the operation
    */
    virtual std::vector<double> bprop(std::vector<ABS_VARIABLE *>& inputs, ABS_VARIABLE & focus, std::vector<double> & gradient) =0;
    /**
     * @brief sets the variable of the operation
    */
    VARIABLE * set_variable(VARIABLE * variable);
};

VARIABLE * OPERATION::set_variable(VARIABLE * variable)
{
    __variable = variable;
    return __variable;
}

#endif // OPERATION_INCLUDE_GUARD