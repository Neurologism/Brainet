#ifndef VOID_OPERATION_INCLUDE_GUARD
#define VOID_OPERATION_INCLUDE_GUARD

#include "operation.h"

/**
 * @brief VOID_OPERATION class is a class for creating an operation that does nothing.
*/
class VOID_OPERATION : public OPERATION
{
public:
    VOID_OPERATION(){};
    void f(std::vector<VARIABLE *>& inputs)override{};
    std::vector<double> bprop(std::vector<VARIABLE *>& inputs, VARIABLE & focus, std::vector<double> & gradient)override{return gradient;};
};

#endif // VOID_OPERATION_INCLUDE_GUARD