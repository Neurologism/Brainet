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

    std::vector<double> __data;
    std::vector<int> __shape;

public:
    VARIABLE(OPERATION * op, std::vector<VARIABLE *> parents, std::vector<VARIABLE *> children) : __op(op), __parents(parents), __children(children){};
    ~VARIABLE();
    OPERATION * get_operation();
    std::vector<VARIABLE *> get_consumers();
    std::vector<VARIABLE *> get_inputs();
    std::vector<double> get_data();
    std::vector<int> get_shape();
    void set_data(std::vector<double> & data);
    void set_shape(std::vector<int> & shape);
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
 * @brief returns the data of the operation
*/
std::vector<double> VARIABLE::get_data()
{
    return __data;
}

/**
 * @brief returns the shape of the data
*/
std::vector<int> VARIABLE::get_shape()
{
    return __shape;
}

/**
 * @brief sets the data of the variable
*/
void VARIABLE::set_data(std::vector<double> & data)
{
    __data = data;
}

/**
 * @brief sets the shape of the data
*/
void VARIABLE::set_shape(std::vector<int> & shape)
{
    __shape = shape;
}

#endif