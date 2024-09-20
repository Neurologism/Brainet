//
// Created by servant-of-scietia on 20.09.24.
//

#include "datatypes/tensor.hpp"

template <class T>
std::uint32_t Tensor<T>::calculateIndex(const ShapeVector &rIndex)
{
    if (rIndex.size() != mShape.size())
        throw std::invalid_argument("Tensor::calculateIndex: Index size does not match the dimensionality of the tensor");

    size_t blockSize = std::accumulate(mShape.begin(), mShape.end(), 1, std::multiplies<>()); // product of all dimensions
    size_t index = 0;

    // calculate the index of the element
    for (std::uint32_t i = 0; i < rIndex.size(); i++)
    {
        blockSize /= mShape[i];
        index += rIndex[i] * blockSize;
    }

    if (index >= mData.size())
        throw std::out_of_range("Index out of range");
    return index;
}

template <class T>
Tensor<T>::Tensor(const ShapeVector &dimensionality)
{
    mData = DataVector(std::accumulate(dimensionality.begin(), dimensionality.end(), 1, std::multiplies<>()), 0);
    mShape = dimensionality;
}

template <class T>
Tensor<T>::Tensor(const ShapeVector &dimensionality, const T &value)
{
    mData = DataVector(std::accumulate(dimensionality.begin(), dimensionality.end(), 1, std::multiplies<>()), value); // initialize the data vector
    mShape = dimensionality;                                                                                                // set the shape of the tensor
}

template <class T>
T Tensor<T>::at(const ShapeVector &index)
{
    return mData[calculateIndex(index)];
}

template <class T>
T Tensor<T>::at(const size_t &index)
{
    if (mData.empty())
        throw std::out_of_range("Tensor::at: Tensor is empty");
    if (index >= mData.size())
        throw std::out_of_range("Index out of range");
    return mData[index];
}

template <class T>
void Tensor<T>::set(const ShapeVector &index, const T &value)
{
    mData[calculateIndex(index)] = value;
}

template <class T>
void Tensor<T>::set(const size_t &index, const T &value)
{
    if (index >= mData.size())
        throw std::out_of_range("Index out of range");
    mData[index] = value;
}

template <class T>
void Tensor<T>::add(const ShapeVector &index, const T &value)
{
    mData[calculateIndex(index)] += value;
}

template <class T>
void Tensor<T>::add(const size_t &index, const T &value)
{
    if (index >= mData.size())
        throw std::out_of_range("Index out of range");
    mData[index] += value;
}

template <class T>
void Tensor<T>::subtract(const ShapeVector &index, const T &value)
{
    mData[calculateIndex(index)] -= value;
}

template <class T>
void Tensor<T>::subtract(const size_t &index, const T &value)
{
    if (index >= mData.size())
        throw std::out_of_range("Index out of range");
    mData[index] -= value;
}

template <class T>
void Tensor<T>::multiply(const ShapeVector &index, const T &value)
{
    mData[calculateIndex(index)] *= value;
}

template <class T>
void Tensor<T>::multiply(const size_t &index, const T &value)
{
    if (index >= mData.size())
        throw std::out_of_range("Index out of range");
    mData[index] *= value;
}

template <class T>
void Tensor<T>::divide(const ShapeVector &index, const T &value)
{
    mData[calculateIndex(index)] /= value;
}

template <class T>
void Tensor<T>::divide(const size_t &index, const T &value)
{
    if (index >= mData.size())
        throw std::out_of_range("Index out of range");
    mData[index] /= value;
}

template <class T>
typename Tensor<T>::ShapeVector Tensor<T>::shape()
{
    return mShape;
}

template <class T>
size_t Tensor<T>::shape(const size_t &index) const
{
    if (index >= mShape.size())
        throw std::out_of_range("Tensor::shape: Index out of range");
    return mShape[index]; // return the shape at the given index
}

template <class T>
std::uint32_t Tensor<T>::dimensionality() const
{
    return mShape.size(); // return the dimensionality
}

template <class T>
std::uint32_t Tensor<T>::capacity()
{
    return mData.size(); // return the capacity
}

template <class T>
void Tensor<T>::resize(const ShapeVector &dimensionality)
{
    mData.resize(std::accumulate(dimensionality.begin(), dimensionality.end(), 1, std::multiplies<>())); // resize the data vector
    mShape = dimensionality;                                                                                   // set the new shape
}

template <class T>
void Tensor<T>::reshape(const ShapeVector &dimensionality)
{
    if (std::accumulate(dimensionality.begin(), dimensionality.end(), 1, std::multiplies<>()) != mData.size())
        throw std::invalid_argument("Tensor::reshape: New dimensionality does not match the capacity of the tensor");
    mShape = dimensionality; // set the new shape
}