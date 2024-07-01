#ifndef MATMUL_INCLUDE_GUARD
#define MATMUL_INCLUDE_GUARD

#include "..\operation.h"



/**
 * @brief Matrix multiplication operation class.
*/
class MATMUL : public OPERATION
{   
    const int block_size = 8; 
    const int threads = 200;
    TENSOR<double> matmul(std::shared_ptr<TENSOR<double>> left_matrix, std::shared_ptr<TENSOR<double>> right_matrix);
    void blockmul(std::shared_ptr<TENSOR<double>> left_matrix, std::shared_ptr<TENSOR<double>> right_matrix, TENSOR<double> & result, int k);
public:
    MATMUL(){};
    void f(std::vector<std::shared_ptr<VARIABLE>>& inputs) override;
    TENSOR<double> bprop(std::vector<std::shared_ptr<VARIABLE>>& inputs, VARIABLE & focus, TENSOR<double> & gradient) override;
};

/**
 * @brief Multithreaded matrix multiplication function by coloum
 * @param left_matrix first matrix
 * @param right_matrix second matrix
 * @param result the matrix that should be returned
 * @param j the current coloum in the result matrix
*/
void MATMUL::blockmul(std::shared_ptr<TENSOR<double>> left_matrix, std::shared_ptr<TENSOR<double>> right_matrix, TENSOR<double> & result, int k)
{
    for (int i = 0; i < left_matrix->shape(0); ++i)
    {
        double sum = 0;
        for (int j = 0; j < left_matrix->shape(1); ++j)
        {
            sum += left_matrix->at({i, j}) * right_matrix->at({j, k});
        }
        result.set({i, k}, sum);
    }
}


/**
 * @brief Matrix multiplication function.
 * @param data1 first matrix
 * @param data2 second matrix
*/
TENSOR<double> MATMUL::matmul(std::shared_ptr<TENSOR<double>> left_matrix, std::shared_ptr<TENSOR<double>> right_matrix)
{
    TENSOR<double> result({left_matrix->shape(0), right_matrix->shape(1)});

    //devide into threads
    std::vector<std::thread> workers(right_matrix->shape(1));
    for (int i = 0; i < right_matrix->shape(1); ++i)
    {
        workers[i] = std::thread (&MATMUL::blockmul, this, left_matrix, right_matrix, std::ref(result), i);
    }
    for (std::thread &worker:workers)
    {
        worker.join();
    }
    
    return result;
}


/**
 * @brief matrix multiplication as operation 
 * handels input and output for the operation and does error checking
 * wraper function for matrix_multiply
*/
void MATMUL::f(std::vector<std::shared_ptr<VARIABLE>>& inputs)
{
    if (inputs.size() != 2)
    {
        throw std::invalid_argument("MATRIX_MULTIPLY::f: Invalid number of input variables.");
    }
    std::shared_ptr<TENSOR<double>> left_matrix = inputs[0]->get_data();
    std::shared_ptr<TENSOR<double>> right_matrix = inputs[1]->get_data();
    if (left_matrix->shape(1)!= right_matrix->shape(0))
    {
        throw std::invalid_argument("MATRIX_MULTIPLY::f: Invalid shapes of input matrices.");
    }
    *(this->get_variable()->get_data()) = matmul(left_matrix, right_matrix);
}


/**
 * @brief backpropagation for matrix multiplication
 * handels input and output for the operation and does error checking
*/
TENSOR<double> MATMUL::bprop(std::vector<std::shared_ptr<VARIABLE>>& inputs, VARIABLE & focus, TENSOR<double> & gradient)
{
    if (inputs.size() != 2)
    {
        throw std::invalid_argument("MATRIX_MULTIPLY::bprop: Invalid number of input variables.");
    }

    std::shared_ptr<TENSOR<double>> left_matrix = inputs[0]->get_data();
    std::shared_ptr<TENSOR<double>> right_matrix = inputs[1]->get_data();

    if(left_matrix->shape(1) != right_matrix->shape(0))
    {
        throw std::invalid_argument("MATRIX_MULTIPLY::bprop: Invalid shapes of input matrices.");
    }


    if (inputs[0]->get_id() == focus.get_id())
    {
        TENSOR<double> right_matrix_transposed = right_matrix->transpose();
        return matmul(std::make_shared<TENSOR<double>>(gradient), std::make_shared<TENSOR<double>>(right_matrix_transposed));
    }
    else 
    {
        TENSOR<double> left_matrix_transposed = left_matrix->transpose();
        return matmul(std::make_shared<TENSOR<double>>(left_matrix_transposed), std::make_shared<TENSOR<double>>(gradient));
    }
}

#endif // MATRX_MULTIPLY_INCLUDE_GUARD