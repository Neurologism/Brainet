#ifndef VARIABLE_INCLUDE_GUARD
#define VARIABLE_INCLUDE_GUARD

#include "dependencies.h"
#include "tensor.h"
#include "operation/operation.h"

class Operation;

/**
 * @brief The variable class is a implementation of a variable in a computational graph. It is used to store the data and owns a pointer to the operation that calculates the data.
*/
class Variable
{
    std::vector<std::shared_ptr<Variable>> __children, __parents; // the children and parents of the variable in the computational graph
    std::shared_ptr<Operation> __op; // the operation that calculates the data
    std::shared_ptr<Tensor<double>> __data; // the data of the variable
    static std::uint32_t __counter; // keep track of the number of variables created
    std::uint32_t __id; // the unique id of the variable
    std::string __operation_name; // the name of the operation that calculates the data

public:

    /**
     * @brief Construct a new Variable object.
     * @param op The operation that calculates the data.
     * @param parents The parents of the variable.
     * @param children The children of the variable.
     * @param data The initial data of the variable.
     */
    Variable(const std::shared_ptr<Operation> & op, const std::vector<std::shared_ptr<Variable>> & parents = {}, const std::vector<std::shared_ptr<Variable>> & children = {}, const std::shared_ptr<Tensor<double>> & data = nullptr);

    /**
     * @brief Copy constructor not allowed.
     */
    Variable(std::shared_ptr<Variable> & var)
    {
        throw std::runtime_error("A variable should not be copied. Use a shared pointer instead.");
    }

    /**
     * @brief Copy assignment not allowed.
     */
    Variable & operator=(std::shared_ptr<Variable> & var)
    {
        throw std::runtime_error("A variable should not be copied. Use a shared pointer instead.");
    }
    Variable(Variable && var) = default;
    Variable & operator=(Variable && var) = default;
    ~Variable() = default;

    /**
     * @brief This function returns the operation that calculates the data.
     * @return std::shared_ptr<Operation> The operation that calculates the data.
     */
    std::shared_ptr<Operation> get_operation();

    /**
     * @brief This function returns the children of the variable.
     * @return std::vector<std::shared_ptr<Variable>> The children of the variable.
     */
    std::vector<std::shared_ptr<Variable>> & get_consumers();

    /**
     * @brief This function returns the parents of the variable.
     * @return std::vector<std::shared_ptr<Variable>> The parents of the variable.
     */
    std::vector<std::shared_ptr<Variable>> & get_inputs();

    /**
     * @brief This function returns the data of the variable.
     * @return std::shared_ptr<Tensor<double>> The data of the variable.
     */
    std::shared_ptr<Tensor<double>> & get_data();

    /**
     * @brief This function returns the id of the variable.
     * @return std::uint32_t The id of the variable.
     */
    std::uint32_t get_id();
};


Variable::Variable(const std::shared_ptr<Operation> & op, const std::vector<std::shared_ptr<Variable>> & parents, const std::vector<std::shared_ptr<Variable>> & children, const std::shared_ptr<Tensor<double>> & data)
{
    __id = __counter++;
    __op = op;
    __parents = parents;
    __children = children;
    __data = data;
    if (op != nullptr)
    {
        __operation_name = op->get_name();
    }
    else
    {
        __operation_name = "INPUT";
    }
};

std::shared_ptr<Operation> Variable::get_operation()
{
    return __op;
}

std::vector<std::shared_ptr<Variable>> & Variable::get_consumers()
{
    return __children;
}

std::vector<std::shared_ptr<Variable>> & Variable::get_inputs()
{
    return __parents;
}

std::shared_ptr<Tensor<double>> & Variable::get_data()
{
    return __data;
}

std::uint32_t Variable::get_id()
{
    return __id;
}

std::uint32_t Variable::__counter = 0; // initialize the static counter

#endif // Variable_INCLUDE_GUARD