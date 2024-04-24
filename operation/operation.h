#ifndef OPERATION_INCLUDE_GUARD
#define OPERATION_INCLUDE_GUARD

#include <vector>
#include "..\variable.h"

/**
 * @brief OPERATION class is an abstract class that defines the interface for all operations.
*/
class OPERATION
{
protected:
    VARIABLE * __variable;
public:
    OPERATION(VARIABLE * variable) : __variable(variable){};
    /**
     * @brief mathematical function the operation implements
    */
    virtual void f(std::vector<VARIABLE *>& inputs) =0; 
    /**
     * @brief derivative of the function
     * assumes that the gradient is already calculated for the output variables 
     * @param inputs the input variables of the operation
     * @param focus the variable to calculate the gradient for
     * @param outputs the output variables of the operation
    */
    virtual void bprop(std::vector<VARIABLE *>& inputs, VARIABLE * focus, std::vector<VARIABLE *> outputs) =0;
};

#endif // OPERATION_INCLUDE_GUARD