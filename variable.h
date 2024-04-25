#ifndef VARIABLE_INCLUDE_GUARD
#define VARIABLE_INCLUDE_GUARD

#include "dependencies.h"
#include "operation/operation.h"

/**
 * @brief VARIABLE class is a wrapper class for OPERATION class. It is used to create a graph of operations.
*/
class VARIABLE
{
    std::vector<VARIABLE *> __children, __parents;
    OPERATION * __op;
    // could use void pointer if required 
    std::vector<double> __data; // tensor of data 
    std::vector<int> __shape; // shape of the tensor 
    int __id;

public:
    VARIABLE(OPERATION * op, std::vector<VARIABLE *> parents, std::vector<VARIABLE *> children) : __op(op), __parents(parents), __children(children){};
    ~VARIABLE();
    OPERATION * get_operation();
    std::vector<VARIABLE *> get_consumers();
    std::vector<VARIABLE *> get_inputs();
    std::vector<double> get_data();
    std::vector<int> get_shape();
    int get_id();
    void set_data(std::vector<double> & data);
    void set_shape(std::vector<int> & shape);
    void set_id(int id);
};

VARIABLE::~VARIABLE()
{
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

/**
 * @brief returns the id of the variable
*/
int VARIABLE::get_id()
{
    return __id;
}

/**
 * @brief sets the id of the variable
*/
void VARIABLE::set_id(int id)
{
    __id = id;
}

#endif