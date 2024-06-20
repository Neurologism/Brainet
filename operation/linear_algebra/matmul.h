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
    void blockmul(TENSOR<double> & data1, TENSOR<double> & data2, TENSOR<double> & result, int j);
public:
    MATMUL(){};
    void f(std::vector<VARIABLE *>& inputs) override;
    TENSOR<double> bprop(std::vector<VARIABLE *>& inputs, VARIABLE & focus, TENSOR<double> & gradient) override;
};

/**
 * @brief Multithreaded matrix multiplication function by coloum
 * @param data1 first matrix
 * @param data2 second matrix
 * @param result the matrix that should be returned
 * @param j the current coloum in the result matrix
*/
void MATMUL::blockmul(TENSOR<double> & data1, TENSOR<double> & data2, TENSOR<double> & result, int j)
{
    for (int i = 0; i < data1.shape()[0]; ++i)
    {
        for (int k = 0; k < data1.shape()[1]; ++k)
        {
            result.set({i,j}, result.at({i,j}) + data1.at({i, k})* data2.at({k, j}));
        }
    }
}


/**
 * @brief Matrix multiplication function.
 * @param data1 first matrix
 * @param data2 second matrix
*/
void MATMUL::matmul(TENSOR<double> & data1, TENSOR<double> & data2)
{
    TENSOR<double> result({data1.shape()[0], data2.shape()[1]});

    //devide into threads
    std::vector<std::thread> workers(data2.shape()[1]);
    for (int i = 0; i < data2.shape()[1]; ++i)
    {
        workers[i] = std::thread (&MATMUL::blockmul, this, std::ref(data1), std::ref(data2), std::ref(result), i);
    }
    for (std::thread &worker:workers)
    {
        worker.join();
    }
    
    data1 = result;
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