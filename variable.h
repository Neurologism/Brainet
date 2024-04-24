#ifndef VARIABLE_INCLUDE_GUARD
#define VARIABLE_INCLUDE_GUARD

#include <vector>
#include "operation/operation.h"
#include "operation/linear_algebra/data_types.h"

/**
 * @brief VARIABLE class is a wrapper class for OPERATION class. It is used to create a graph of operations.
*/
class VARIABLE
{
    std::vector<VARIABLE *> __children, __parents;
    OPERATION * __op;
    std::vector<DATATYPE *> __data;
    

public:
    VARIABLE(OPERATION * op, std::vector<VARIABLE *> parents, std::vector<VARIABLE *> children) : __op(op), __parents(parents), __children(children){};
    ~VARIABLE();
    OPERATION * get_operation();
    std::vector<VARIABLE *> get_consumers();
    std::vector<VARIABLE *> get_inputs();
    DATATYPE * get_data();
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

/**
 * @brief returns the data of the variable
*/
DATATYPE * VARIABLE::get_data()
{
    return __data[0];
}

#endif