#ifndef MATMUL_INCLUDE_GUARD
#define MATMUL_INCLUDE_GUARD

#include <vector>
#include <stdexcept>
#include "..\operation.h"

/**
 * @brief Matrix multiplication operation class.
*/
class MATMUL : public OPERATION
{
public:
    MATMUL(VARIABLE * variable) : OPERATION(variable){};
    void f(std::vector<VARIABLE *>& inputs) override;
    void bprop(std::vector<VARIABLE *>& inputs, std::vector<VARIABLE *> outputs) override;
    void matmul(TENSOR * data1, TENSOR * data2, TENSOR * result);
};


/**
 * @brief Matrix multiplication function.
 * @attention to be replaced 
*/
void MATMUL::matmul(TENSOR * data1, TENSOR * data2, TENSOR * result)
{
    for()
}


/**
 * @brief matrix multiplication as operation 
 * handels input and output for the operation and does error checking
 * wraper function for matrix_multiply
*/
void MATMUL::f(std::vector<VARIABLE *>& inputs)
{
    if (inputs.size() != 2)
    {
        throw std::invalid_argument("MATRIX_MULTIPLY::f: Invalid number of input variables.");
    }

    std::vector<int> shape1 = inputs[0]->get_shape();
    std::vector<int> shape2 = inputs[1]->get_shape();

    if (shape1[1] != shape2[0])
    {
        throw std::invalid_argument("MATRIX_MULTIPLY::f: Invalid shapes of input matrices.");
    }

    std::vector<double> data1 = inputs[0]->get_data();
    std::vector<double> data2 = inputs[1]->get_data();

    matmul(data1, data2, shape1, shape2);

    __variable->set_data(data1);
    __variable->set_shape(shape1);

}


/**
 * @brief backpropagation for matrix multiplication
 * handels input and output for the operation and does error checking
*/
void MATMUL::bprop(std::vector<VARIABLE *>& inputs, std::vector<VARIABLE *> outputs)
{
    if (inputs.size() != 2)
    {
        throw std::invalid_argument("MATRIX_MULTIPLY::bprop: Invalid number of input variables.");
    }

    std::vector<int> shape1 = inputs[0]->get_shape();
    std::vector<int> shape2 = inputs[1]->get_shape();

    if (shape1[1] != shape2[0])
    {
        throw std::invalid_argument("MATRIX_MULTIPLY::bprop: Invalid shapes of input matrices.");
    }

    std::vector<double> data1 = inputs[0]->get_data();
    std::vector<double> data2 = inputs[1]->get_data();




}

#endif // MATRX_MULTIPLY_INCLUDE_GUARD