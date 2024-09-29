//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/matmul.hpp"

void Matmul::blockmul(const std::shared_ptr<Matrix> &left_matrix, const std::shared_ptr<Matrix> &right_matrix, std::shared_ptr<Matrix> &result, const std::uint32_t &k)
{
    // straight forward matrix vector multiplication
    for (std::uint32_t i = 0; i < left_matrix->shape(0); ++i)
    {
        Precision sum = 0;
        for (std::uint32_t j = 0; j < left_matrix->shape(1); ++j)
        {
            sum += left_matrix->at(i, j) * right_matrix->at(j, k);
        }
        result->set(i, k, sum);
    }
}

void Matmul::matmul(std::shared_ptr<Matrix> left_matrix, std::shared_ptr<Matrix> right_matrix, std::shared_ptr<Matrix> result)
{
    //divide into threads
    std::vector<std::thread> workers(right_matrix->shape(1));
    for (std::uint32_t i = 0; i < right_matrix->shape(1); i++)
    {
        workers[i] = std::thread (&Matmul::blockmul, this, std::ref(left_matrix), std::ref(right_matrix), std::ref(result), i);
    }
    for (std::thread &worker:workers)
    {
        worker.join(); // wait for all threads to finish
    }
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
    std::shared_ptr<Matrix> left_matrix = std::static_pointer_cast<Matrix>(inputs[0]->getData());
    std::shared_ptr<Matrix> right_matrix = std::static_pointer_cast<Matrix>(inputs[1]->getData());
    if (this->getVariable()->getData() == nullptr || this->getVariable()->getData()->capacity() != left_matrix->shape(0) * right_matrix->shape(1))
    {
        this->getVariable()->setData(std::make_shared<Matrix>(Matrix({left_matrix->shape(0), right_matrix->shape(1)}, 0)));
    }
    std::shared_ptr<Matrix> result = std::static_pointer_cast<Matrix>(this->getVariable()->getData());
    matmul(left_matrix, right_matrix, result);
}

std::shared_ptr<Tensor> Matmul::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor> & gradient)
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
    std::shared_ptr<Matrix> gradient_matrix = static_pointer_cast<Matrix>(gradient);
    if (inputs[0]->getId() == focus->getId())
    {
        std::shared_ptr<Matrix> right_matrix_transposed = static_cast<Matrix*>(inputs[1]->getData().get())->transpose(); // transposed version needed to output the correct shape
        std::shared_ptr<Matrix> result = std::make_shared<Matrix>(inputs[0]->getData()->shape(), 0);
        matmul(gradient_matrix, right_matrix_transposed, result);
        return result;
    }
    else
    {
        std::shared_ptr<Matrix> left_matrix_transposed = static_cast<Matrix*>(inputs[0]->getData().get())->transpose(); // transposed version needed to output the correct shape
        std::shared_ptr<Matrix> result = std::make_shared<Matrix>(inputs[1]->getData()->shape(), 0);
        matmul(left_matrix_transposed, gradient_matrix, result);
        return result;
    }
}