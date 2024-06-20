#ifndef MATMUL_INCLUDE_GUARD
#define MATMUL_INCLUDE_GUARD

#include "..\operation.h"

const int block_size = 8; 
const int threads = 200;

/**
 * @brief Matrix multiplication operation class.
*/
class MATMUL : public OPERATION
{
    void matmul(TENSOR<double> & data1, TENSOR<double> & data2);
    //void blockmul(std::vector<double> & data1, std::vector<double> & data2, std::vector<double> & result, std::vector<int> &shape1, std::vector<int> &shape2, int start, int ende);
    void blockmul(TENSOR<double> & data1, TENSOR<double> & data2, std::vector<double> & result, std::vector<int> &shape1, std::vector<int> &shape2, int j);
public:
    MATMUL(){};
    void f(std::vector<VARIABLE *>& inputs) override;
    std::TENSOR<double> bprop(std::vector<VARIABLE *>& inputs, VARIABLE & focus, TENSOR<double> & gradient) override;
};


void MATMUL::blockmul(std::vector<double> & data1, std::vector<double> & data2, std::vector<double> & result, std::vector<int> & shape1, std::vector<int> & shape2, int j)
{
    for (int i = 0; i < shape1[0]; ++i)
    {
        for (int k = 0; k < shape1[1]; ++k)
        {
            result[i*shape2[1]+j] += data1[i * shape1[1] + k] * data2[k * shape2[1] + j];
        }
    }
}


/**
 * @brief Matrix multiplication function.
*/
void MATMUL::matmul(std::vector<double> & data1, std::vector<double> & data2, std::vector<int> & shape1, std::vector<int> & shape2)
{
    std::vector<double> result(shape1[0]*shape2[1], 0);

    //devide into threads
    std::vector<std::thread> workers(shape2[1]);
    for (int i = 0; i < shape2[1]; ++i)
    {
        workers[i] = std::thread (&MATMUL::blockmul, this, std::ref(data1), std::ref(data2), std::ref(result), std::ref(shape1), std::ref(shape2), i);
    }
    for (std::thread &worker:workers)
    {
        worker.join();
    }
    
    data1.swap(result);
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
std::vector<double> MATMUL::bprop(std::vector<VARIABLE *>& inputs, VARIABLE & focus, std::vector<double> & gradient)
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

    std::vector<double> data;

    if (inputs[0]->get_id() != focus.get_id())
    {
        data = inputs[1]->get_data();
        shape1.swap(shape2);
    }
    else data = inputs[0]->get_data();



    matmul(data, gradient, shape1, shape2);

    return data;
}

#endif // MATRX_MULTIPLY_INCLUDE_GUARD