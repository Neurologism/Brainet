#ifndef MATMUL_HPP
#define MATMUL_HPP

#include "operation.hpp"


/**
 * @brief Matmul class used to perform the dot product of two matrices. This is usually a bottleneck in neural networks. 
 * Expecting to replace this with a more efficient CUDA implementation in the future.
*/
class Matmul : public Operation
{   
protected:
    static const std::uint32_t threads = 200;
    /**
     * @brief  matrix vector multiplication function
     * @param left_matrix the left matrix
     * @param right_matrix the right matrix
     * @param result the result of the matrix multiplication
     * @param k the index of the coloum in the right matrix
     */
    void blockmul(Matrix &left_matrix, Matrix &right_matrix, Matrix &result, const std::uint32_t &k, const bool &left_transpose, const bool &right_transpose);

    /**
     * @brief spawning threads for every coloum in the right matrix to execute the blockmul function in parallel
     * @param left_matrix the left matrix
     * @param right_matrix the right matrix
     * @param result the result of the matrix multiplication
     */
    void matmul(const std::shared_ptr<Matrix>& left_matrix, const std::shared_ptr<Matrix>& right_matrix, const std::shared_ptr<Matrix>& result, const bool &left_transpose = false, const bool &right_transpose = false);
public:    
    Matmul(){mName = "Matmul";};
    ~Matmul(){};
    /**
     * @brief wrapper function for matmul. Does error checking and handles inputs and outputs.
     * @param inputs the input variables
     */
    void f(std::vector<std::shared_ptr<Variable>>& inputs) override;
    /**
     * @brief bprop for matmul. Outputs the gradient multiplied by the input != focus
     * @param inputs the input variables
     * @param focus the variable to calculate the gradient for
     * @param gradient the sum of the gradients of the consumers
     */
    std::shared_ptr<Tensor> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor> & gradient) override;
};

#endif // MATMUL_HPP