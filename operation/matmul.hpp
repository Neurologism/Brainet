#ifndef MATMUL_HPP
#define MATMUL_HPP

#include "operation.hpp"


/**
 * @brief Matmul class used to perform the dot product of two matrices. This is usually a bottleneck in neural networks. 
 * Expecting to replace this with a more efficient implementation in the future. Credits to @s1m-ba for the implementation.
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
    void blockmul(std::shared_ptr<Tensor<double>> & left_matrix, std::shared_ptr<Tensor<double>> & right_matrix, std::shared_ptr<Tensor<double>> & result, std::uint32_t k);

    /**
     * @brief spawning threads for every coloum in the right matrix to execute the blockmul function in parallel
     * @param left_matrix the left matrix
     * @param right_matrix the right matrix
     * @return the result of the matrix multiplication
     */
    std::shared_ptr<Tensor<double>> matmul(std::shared_ptr<Tensor<double>> & left_matrix, std::shared_ptr<Tensor<double>> & right_matrix);
public:    
    Matmul(){__dbg_name = "Matmul";};
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
    std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient) override;
};

void Matmul::blockmul(std::shared_ptr<Tensor<double>> & left_matrix, std::shared_ptr<Tensor<double>> & right_matrix, std::shared_ptr<Tensor<double>> & result, std::uint32_t k)
{
    // straight forward matrix vector multiplication
    for (std::uint32_t i = 0; i < left_matrix->shape(0); ++i)
    {
        double sum = 0;
        for (std::uint32_t j = 0; j < left_matrix->shape(1); ++j)
        {
            sum += left_matrix->at({i, j}) * right_matrix->at({j, k});
        }
        result->set({i, k}, sum);
    }
}

std::shared_ptr<Tensor<double>> Matmul::matmul(std::shared_ptr<Tensor<double>> & left_matrix, std::shared_ptr<Tensor<double>> & right_matrix)
{
    std::shared_ptr<Tensor<double>> result = std::make_shared<Tensor<double>>(Tensor<double>({left_matrix->shape(0),right_matrix->shape(1)}));

    //devide into threads
    std::vector<std::thread> workers(right_matrix->shape(1));
    for (std::uint32_t i = 0; i < right_matrix->shape(1); ++i)
    {
        workers[i] = std::thread (&Matmul::blockmul, this, std::ref(left_matrix), std::ref(right_matrix), std::ref(result), i);
    }
    for (std::thread &worker:workers)
    {
        worker.join(); // wait for all threads to finish
    }
    
    return result;
}

void Matmul::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    // error checking
    if (inputs.size() != 2)
    {
        throw std::invalid_argument("MATRIX_MULTIPLY::f: Invalid number of input variables.");
    }
    ;
    if (inputs[0]->get_data()->shape(1)!= inputs[1]->get_data()->shape(0))
    {
        throw std::invalid_argument("MATRIX_MULTIPLY::f: Invalid shapes of input matrices.");
    }
    // perform the matrix multiplication
    this->get_variable()->get_data() = matmul(inputs[0]->get_data(), inputs[1]->get_data());
}

std::shared_ptr<Tensor<double>> Matmul::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient)
{
    if (inputs.size() != 2)
    {
        throw std::invalid_argument("MATRIX_MULTIPLY::bprop: Invalid number of input variables.");
    }
    if(inputs[0]->get_data()->shape(1) != inputs[1]->get_data()->shape(0))
    {
        throw std::invalid_argument("MATRIX_MULTIPLY::bprop: Invalid shapes of input matrices.");
    }
    

    // return the gradient multiplied by the input != focus
    if (inputs[0]->get_id() == focus->get_id())
    {
        std::shared_ptr<Tensor<double>> right_matrix_transposed = static_cast<Matrix<double>*>(inputs[1]->get_data().get())->transpose(); // transposed version needed to output the correct shape
        return matmul(gradient, right_matrix_transposed);
    }
    else 
    {
        std::shared_ptr<Tensor<double>> left_matrix_transposed = static_cast<Matrix<double>*>(inputs[0]->get_data().get())->transpose(); // transposed version needed to output the correct shape
        return matmul(left_matrix_transposed, gradient);
    }
}

#endif // MATMUL_HPP