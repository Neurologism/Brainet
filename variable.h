#ifndef VARIABLE_INCLUDE_GUARD
#define VARIABLE_INCLUDE_GUARD

#include "dependencies.h"
#include "tensor.h"
#include "operation/operation.h"

class OPERATION;

/**
 * @brief VARIABLE class is a wrapper class for OPERATION class. It is used to create a graph of operations.
*/
class VARIABLE
{
    std::vector<std::shared_ptr<VARIABLE>> __children, __parents;
    std::shared_ptr<OPERATION> __op;
    std::shared_ptr<TENSOR<double>> __data; // each variable can store a tensor of the data 
    static int __counter; // keep track of the number of variables created
    int __id;

public:
    VARIABLE(std::shared_ptr<OPERATION> op)
    {
        __id = __counter++;
        __op = op;
    };
    VARIABLE(std::shared_ptr<OPERATION> op, std::vector<std::shared_ptr<VARIABLE>> parents, std::vector<std::shared_ptr<VARIABLE>> children)
    {
        __id = __counter++;
        __op = op;
        __parents = parents;
        __children = children;
    };
    VARIABLE(std::shared_ptr<OPERATION> op, std::vector<std::shared_ptr<VARIABLE>> parents, std::vector<std::shared_ptr<VARIABLE>> children, TENSOR<double> & data)
    {
        __id = __counter++;
        __op = op;
        __parents = parents;
        __children = children;
        *__data = data;
    };
    std::shared_ptr<OPERATION> get_operation();
    std::vector<std::shared_ptr<VARIABLE>> get_consumers();
    std::vector<std::shared_ptr<VARIABLE>> get_inputs();
    std::shared_ptr<TENSOR<double>> get_data();
    int get_id();
};

/**
 * @brief returns the operation object
*/
std::shared_ptr<OPERATION> VARIABLE::get_operation()
{
    return __op;
}

/**
 * @brief returns the children of the variable
*/
std::vector<std::shared_ptr<VARIABLE>> VARIABLE::get_consumers()
{
    return __children;
}

/**
 * @brief returns the parents of the variable
*/
std::vector<std::shared_ptr<VARIABLE>> VARIABLE::get_inputs()
{
    return __parents;
}

/**
 * @brief returns the data of the operation
*/
std::shared_ptr<TENSOR<double>> VARIABLE::get_data()
{
    return __data;
}

/**
 * @brief returns the id of the variable
*/
int VARIABLE::get_id()
{
    return __id;
}

int VARIABLE::__counter = 0;

#endif