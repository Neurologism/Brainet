//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/matmul.hpp"

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
    if (inputs[0]->getData()->shape(1)!= inputs[1]->getData()->shape(0))
    {
        throw std::invalid_argument("MATRIX_MULTIPLY::f: Invalid shapes of input matrices.");
    }
    // perform the matrix multiplication
    this->getVariable()->getData() = matmul(inputs[0]->getData(), inputs[1]->getData());
}

std::shared_ptr<Tensor<double>> Matmul::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient)
{
    if (inputs.size() != 2)
    {
        throw std::invalid_argument("MATRIX_MULTIPLY::bprop: Invalid number of input variables.");
    }
    if(inputs[0]->getData()->shape(1) != inputs[1]->getData()->shape(0))
    {
        throw std::invalid_argument("MATRIX_MULTIPLY::bprop: Invalid shapes of input matrices.");
    }


    // return the gradient multiplied by the input != focus
    if (inputs[0]->getId() == focus->getId())
    {
        std::shared_ptr<Tensor<double>> right_matrix_transposed = static_cast<Matrix<double>*>(inputs[1]->getData().get())->transpose(); // transposed version needed to output the correct shape
        return matmul(gradient, right_matrix_transposed);
    }
    else
    {
        std::shared_ptr<Tensor<double>> left_matrix_transposed = static_cast<Matrix<double>*>(inputs[0]->getData().get())->transpose(); // transposed version needed to output the correct shape
        return matmul(left_matrix_transposed, gradient);
    }
}