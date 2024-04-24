#ifndef MATMUL_INCLUDE_GUARD
#define MATMUL_INCLUDE_GUARD

#include <vector>
#include <stdexcept>
#include <array>
#include "..\operation.h"

/**
 * @brief Matrix multiplication operation class.
*/
class MATMUL : public OPERATION
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
void MATMUL::matmul(std::vector<double> & data1, std::vector<double> & data2, std::vector<int> & shape1, std::vector<int> & shape2)
{
    std::vector<double> result;

    for (int i = 0; i < shape1[0]; i++) // replace this 
    {
        for (int j = 0; j < shape2[1]; j++)
        {
            double sum = 0;
            for (int k = 0; k < shape1[1]; k++)
            {
                sum += data1[i * shape1[1] + k] * data2[k * shape2[1] + j];
            }
            result.push_back(sum);
        }
    }

    data1 = result;
    shape1 = {shape1[0], shape2[1]};
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