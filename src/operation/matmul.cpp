//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/matmul.hpp"

void Matmul::blockmul(Matrix &left_matrix, Matrix &right_matrix, Matrix &result, const std::uint32_t &k, const bool &left_transpose, const bool &right_transpose)
{
    std::vector<float> &left_data = left_matrix.getData();
    std::vector<float> &right_data = right_matrix.getData();
    std::vector<size_t> &left_shape = left_matrix.getShape();
    std::vector<size_t> &right_shape = right_matrix.getShape();
    std::uint32_t left_index = 0;
    std::uint32_t right_index = 0;
    if (!left_transpose && !right_transpose)
    {
        for (std::uint32_t i = 0; i < left_matrix.shape(0); ++i)
        {
            left_index = i * left_shape[1];
            right_index = k;
            Precision sum = 0;
            const std::uint32_t shape = left_matrix.shape(1);
            const std::uint32_t right_stride = right_shape[1];
            for (std::uint32_t j = 0; j < shape; ++j)
            {
                sum += left_data[left_index] * right_data[right_index];
                left_index++;
                right_index += right_stride;
            }
            result.set(i, k, sum);
        }
    }
    else if (left_transpose && !right_transpose)
    {
        for (std::uint32_t i = 0; i < left_matrix.shape(1); ++i)
        {
            left_index = i;
            right_index = k;
            Precision sum = 0;
            const std::uint32_t shape = left_matrix.shape(0);
            const std::uint32_t left_stride = left_shape[1];
            const std::uint32_t right_stride = right_shape[1];
            for (std::uint32_t j = 0; j < shape; ++j)
            {
                sum += left_data[left_index] * right_data[right_index];
                left_index += left_stride;
                right_index += right_stride;
            }
            result.set(i, k, sum);
        }
    }
    else if (!left_transpose && right_transpose)
    {
        for (std::uint32_t i = 0; i < left_matrix.shape(0); ++i)
        {
            left_index = i * left_shape[1];
            right_index = k * right_shape[0];
            Precision sum = 0;
            const std::uint32_t shape = left_matrix.shape(1);
            for (std::uint32_t j = 0; j < shape; ++j)
            {
                sum += left_data[left_index] * right_data[right_index];
                left_index++;
                right_index++;
            }
            result.set(i, k, sum);
        }
    }
    else
    {
        for (std::uint32_t i = 0; i < left_matrix.shape(0); ++i)
        {
            left_index = i;
            right_index = k * right_shape[0];
            Precision sum = 0;
            const std::uint32_t shape = left_matrix.shape(1);
            const std::uint32_t left_stride = left_shape[1];
            for (std::uint32_t j = 0; j < shape; ++j)
            {
                sum += left_data[left_index] * right_data[right_index];
                left_index += left_stride;
                right_index++;
            }
            result.set(i, k, sum);
        }
    }
}

void Matmul::matmul(const std::shared_ptr<Matrix>& left_matrix, const std::shared_ptr<Matrix>& right_matrix, const std::shared_ptr<Matrix>& result, const bool &left_transpose, const bool &right_transpose)
{
    //divide into threads
    std::vector<std::thread> workers(right_matrix->shape(1));
    Matrix &left_matrix_ref = *left_matrix;
    Matrix &right_matrix_ref = *right_matrix;
    Matrix &result_ref = *result;
    for (std::uint32_t i = 0; i < right_matrix->shape(1); i++)
    {
        workers[i] = std::thread (&Matmul::blockmul, this, std::ref(left_matrix_ref), std::ref(right_matrix_ref), std::ref(result_ref), i, std::ref(left_transpose), std::ref(right_transpose));
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
        std::shared_ptr<Matrix> right_matrix = std::static_pointer_cast<Matrix>(inputs[1]->getData()); // transposed version needed to output the correct shape
        std::shared_ptr<Matrix> result = std::make_shared<Matrix>(inputs[0]->getData()->shape(), 0);
        matmul(gradient_matrix, right_matrix, result, false, true);
        return result;
    }
    else
    {
        std::shared_ptr<Matrix> left_matrix = std::static_pointer_cast<Matrix>(inputs[0]->getData()); // transposed version needed to output the correct shape
        std::shared_ptr<Matrix> result = std::make_shared<Matrix>(inputs[1]->getData()->shape(), 0);
        matmul(left_matrix, gradient_matrix, result, true, false);
        return result;
    }
}