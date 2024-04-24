#ifndef MATMUL_INCLUDE_GUARD
#define MATMUL_INCLUDE_GUARD

#include <vector>
#include <stdexcept>
#include <array>
#include "..\operation.h"

/**
 * @brief Matrix multiplication operation class.
*/
template<typename T>
class MATMUL : public OPERATION<T>
{
public:

    MATMUL(VARIABLE * variable) : OPERATION(variable){};
    void f(std::vector<VARIABLE *>& inputs) override;
    std::vector<double> bprop(std::vector<VARIABLE *>& inputs, VARIABLE * focus, std::vector<VARIABLE *> outputs) override;
    void matmul(std::vector<double> & data1, std::vector<double> & data2, std::vector<int> & shape1, std::vector<int> & shape2);

};


/**
 * @brief Matrix multiplication function.
 * @attention to be replaced 
*/
template<typename T>
void MATMUL<T>::matmul(T * data1, T * data2, T * result)
{
    for()
}


/**
 * @brief matrix multiplication as operation 
 * handels input and output for the operation and does error checking
 * wraper function for matrix_multiply
*/
template<typename T>
void MATMUL<T>::f(std::vector<VARIABLE<T> *>& inputs)
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

std::vector<double> MATMUL::bprop(std::vector<VARIABLE *>& inputs, VARIABLE * focus, std::vector<VARIABLE *> outputs)
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