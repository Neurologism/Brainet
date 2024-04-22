#ifndef VARIABLE_INCLUDE_GUARD
#define VARIABLE_INCLUDE_GUARD

#include <vector>
#include "layer/operation.h"

/**
 * @brief VARIABLE class is a wrapper class for OPERATION class. It is used to create a graph of operations.
*/
class VARIABLE
{
    std::vector<VARIABLE *> __children, __parents;
    OPERATION * __op;

public:
    VARIABLE(OPERATION * op, std::vector<VARIABLE *> parents, std::vector<VARIABLE *> children) : __op(op), __parents(parents), __children(children){};
    ~VARIABLE();
    OPERATION * get_operation();
    std::vector<VARIABLE *> get_consumers();
    std::vector<VARIABLE *> get_inputs();
};

VARIABLE::~VARIABLE()
{
    free(__op);
    for (VARIABLE * parent : __parents)
    {
        free(parent);
    }
    for (VARIABLE * child : __children)
    {
        free(child);
    }
}

/**
 * @brief returns the operation object
*/
OPERATION * VARIABLE::get_operation()
{
    return __op;
}

/**
 * @brief returns the children of the variable
*/
std::vector<VARIABLE *> VARIABLE::get_consumers()
{
    return __children;
}

/**
 * @brief returns the parents of the variable
*/
std::vector<VARIABLE *> VARIABLE::get_inputs()
{
    return __parents;
}


#endif