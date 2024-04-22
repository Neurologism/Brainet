#ifndef OPERATION_INCLUDE_GUARD
#define OPERATION_INCLUDE_GUARD

#include <vector>

/**
 * @brief OPERATION class is an abstract class that defines the interface for all operations.
*/
class OPERATION
{

public:
    OPERATION(){};
    virtual void f() =0; 
    virtual void bprop() =0;
};

#endif // OPERATION_INCLUDE_GUARD