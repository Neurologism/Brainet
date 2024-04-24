#ifndef OPERATION_INCLUDE_GUARD
#define OPERATION_INCLUDE_GUARD

#include <vector>
#include "..\variable.h"

/**
 * @brief OPERATION class is an abstract class that defines the interface for all operations.
*/
template <typename T>
class OPERATION
{
public:
    /**
     * @brief mathematical function the operation implements
    */
    virtual void f(std::vector<T *>& inputs) =0; 
    /**
     * @brief derivative of the function
     * assumes that the gradient is already calculated for the output variables 
     * gives space to storage optmisations as we are computing the gradient for all inputs at once
     * @param inputs the input variables of the operation
     * @param outputs the output variables of the operation
    */
    virtual void bprop(std::vector<T *>& inputs, std::vector<T *> outputs) =0;
};

#endif // OPERATION_INCLUDE_GUARD