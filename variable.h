#ifndef VARIABLE_INCLUDE_GUARD
#define VARIABLE_INCLUDE_GUARD

#include "dependencies.h"
#include "tensor.h"
#include "operation/operation.h"

class OPERATION;

/**
 * @brief The variable class is a implementation of a variable in a computational graph. It is used to store the data and owns a pointer to the operation that calculates the data.
*/
class VARIABLE
{
    std::vector<std::shared_ptr<VARIABLE>> __children, __parents; // the children and parents of the variable in the computational graph
    std::shared_ptr<OPERATION> __op; // the operation that calculates the data
    std::shared_ptr<TENSOR<double>> __data; // the data of the variable
    static std::uint32_t __counter; // keep track of the number of variables created
    std::uint32_t __id; // the unique id of the variable

public:
    /**
     * @brief Construct a new VARIABLE object.
     * @param op The operation that calculates the data.
     * @param parents The parents of the variable.
     * @param children The children of the variable.
     * @param data The initial data of the variable.
     */
    VARIABLE(const std::shared_ptr<OPERATION> & op, const std::vector<std::shared_ptr<VARIABLE>> & parents = {}, const std::vector<std::shared_ptr<VARIABLE>> & children = {}, const std::shared_ptr<TENSOR<double>> & data = nullptr);
    /**
     * @brief Copy constructor not allowed.
     */
    VARIABLE(std::shared_ptr<VARIABLE> & var)
    {
        throw std::runtime_error("A variable should not be copied. Use a shared pointer instead.");
    }
    /**
     * @brief Copy assignment not allowed.
     */
    VARIABLE & operator=(std::shared_ptr<VARIABLE> & var)
    {
        throw std::runtime_error("A variable should not be copied. Use a shared pointer instead.");
    }
    VARIABLE(VARIABLE && var) = default;
    VARIABLE & operator=(VARIABLE && var) = default;
    ~VARIABLE() = default;
    /**
     * @brief This function returns the operation that calculates the data.
     * @return std::shared_ptr<OPERATION> The operation that calculates the data.
     */
    std::shared_ptr<OPERATION> get_operation();
    /**
     * @brief This function returns the children of the variable.
     * @return std::vector<std::shared_ptr<VARIABLE>> The children of the variable.
     */
    std::vector<std::shared_ptr<VARIABLE>> & get_consumers();
    /**
     * @brief This function returns the parents of the variable.
     * @return std::vector<std::shared_ptr<VARIABLE>> The parents of the variable.
     */
    std::vector<std::shared_ptr<VARIABLE>> & get_inputs();
    /**
     * @brief This function returns the data of the variable.
     * @return std::shared_ptr<TENSOR<double>> The data of the variable.
     */
    std::shared_ptr<TENSOR<double>> & get_data();
    /**
     * @brief This function returns the id of the variable.
     * @return std::uint32_t The id of the variable.
     */
    std::uint32_t get_id();
};


VARIABLE::VARIABLE(const std::shared_ptr<OPERATION> & op, const std::vector<std::shared_ptr<VARIABLE>> & parents, const std::vector<std::shared_ptr<VARIABLE>> & children, const std::shared_ptr<TENSOR<double>> & data)
{
    __id = __counter++;
    __op = op;
    __parents = parents;
    __children = children;
    __data = data;
};

std::shared_ptr<OPERATION> VARIABLE::get_operation()
{
    return __op;
}

std::vector<std::shared_ptr<VARIABLE>> & VARIABLE::get_consumers()
{
    return __children;
}

std::vector<std::shared_ptr<VARIABLE>> & VARIABLE::get_inputs()
{
    return __parents;
}

std::shared_ptr<TENSOR<double>> & VARIABLE::get_data()
{
    return __data;
}

std::uint32_t VARIABLE::get_id()
{
    return __id;
}

std::uint32_t VARIABLE::__counter = 0; // initialize the static counter

#endif // VARIABLE_INCLUDE_GUARD