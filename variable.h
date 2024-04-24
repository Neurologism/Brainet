#ifndef VARIABLE_INCLUDE_GUARD
#define VARIABLE_INCLUDE_GUARD

#include <vector>
#include "operation/operation.h"

/**
 * @brief VARIABLE class is a wrapper class for OPERATION class. It is used to create a graph of operations.
*/
template <typename T>
class VARIABLE
{
    std::vector<VARIABLE *> __children, __parents;
    OPERATION<VARIABLE<T>> __op;
    T __data;
    

public:
    VARIABLE(OPERATION * op, std::vector<VARIABLE *> parents, std::vector<VARIABLE *> children) : __op(op), __parents(parents), __children(children){};
    ~VARIABLE();
    OPERATION<VARIABLE<T>> * get_operation();
    std::vector<VARIABLE *> get_consumers();
    std::vector<VARIABLE *> get_inputs();
};

template <typename T>
VARIABLE<T>::~VARIABLE()
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
template <typename T>
OPERATION<VARIABLE<T>> * VARIABLE<T>::get_operation()
{
    return __op;
}

/**
 * @brief returns the children of the variable
*/
template <typename T>
std::vector<VARIABLE<T> *> VARIABLE<T>::get_consumers()
{
    return __children;
}

/**
 * @brief returns the parents of the variable
*/
template <typename T>
std::vector<VARIABLE<T> *> VARIABLE<T>::get_inputs()
{
    return __parents;
}

#endif