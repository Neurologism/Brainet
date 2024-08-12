#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "tensor.hpp"

/**
 * @brief The Matrix class is a wrapper around the Tensor class that provides an extended interface for matrix operations.
 */
template <typename T>
class Matrix : public Tensor<T>
{


public:

    std::shared_ptr<Matrix<T>> transpose();
};

template <class T>
std::shared_ptr<Matrix<T>> Matrix<T>::transpose()
{
    if(mShape.size() != 2)
        throw std::invalid_argument("Tensor::transpose: Tensor must be 2D");
    std::shared_ptr<Tensor<T>> _tensor = std::make_shared<Tensor<T>>(Tensor<T>({mShape[1],mShape[0]})); // create a new tensor with the transposed shape
    DataVector _data(mData.size());
    // create a vector with the transposed data
    for(ShapeType i = 0; i < mShape[0]; i++)
    {
        for(ShapeType j = 0; j < mShape[1]; j++)
        {
            _data[j*mShape[0] + i] = mData[i*mShape[1] + j];
        }
    }
    _tensor->data() = _data; // set the data of the new tensor
    return _tensor; // return the new tensor
}

#endif // MATRIX_HPP