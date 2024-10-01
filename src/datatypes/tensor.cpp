//
// Created by servant-of-scietia on 9/28/24.
//

#include "datatypes/tensor.hpp"

std::uint32_t Tensor::calculateIndex(const ShapeVector &rIndex)
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

Tensor::Tensor(const ShapeVector &dimensionality)
{
    mData = DataVector(std::accumulate(dimensionality.begin(), dimensionality.end(), 1, std::multiplies<>()), 0);
    mShape = dimensionality;
}

Tensor::Tensor(const ShapeVector &dimensionality, const Precision &value)
{
    mData = DataVector(std::accumulate(dimensionality.begin(), dimensionality.end(), 1, std::multiplies<>()), value); // initialize the data vector
    mShape = dimensionality;                                                                                                // set the shape of the tensor
}

Tensor::Tensor(const Tensor &tensor)
{
    mData = tensor.mData; // copy the data
    mShape = tensor.mShape; // copy the shape
}

Tensor& Tensor::operator=(const Tensor &tensor)
{
    if (this == &tensor)
        return *this;
    mData = tensor.mData; // copy the data
    mShape = tensor.mShape; // copy the shape
    return *this;
}

Precision Tensor::at(const ShapeVector &index)
{
    return mData[calculateIndex(index)];
}

Precision Tensor::at(const size_t &index)
{
    return mData[index];
}

void Tensor::set(const ShapeVector &index, const Precision &value)
{
    mData[calculateIndex(index)] = value;
}

void Tensor::set(const size_t &index, const Precision &value)
{
    if (index >= mData.size())
        throw std::out_of_range("Index out of range");
    mData[index] = value;
}

void Tensor::add(const ShapeVector &index, const Precision &value)
{
    mData[calculateIndex(index)] += value;
}

void Tensor::add(const size_t &index, const Precision &value)
{
    if (index >= mData.size())
        throw std::out_of_range("Index out of range");
    mData[index] += value;
}

void Tensor::subtract(const ShapeVector &index, const Precision &value)
{
    mData[calculateIndex(index)] -= value;
}

void Tensor::subtract(const size_t &index, const Precision &value)
{
    mData[index] -= value;
}

void Tensor::multiply(const ShapeVector &index, const Precision &value)
{
    mData[calculateIndex(index)] *= value;
}

void Tensor::multiply(const size_t &index, const Precision &value)
{
    if (index >= mData.size())
        throw std::out_of_range("Index out of range");
    mData[index] *= value;
}

void Tensor::divide(const ShapeVector &index, const Precision &value)
{
    mData[calculateIndex(index)] /= value;
}

void Tensor::divide(const size_t &index, const Precision &value)
{
    if (index >= mData.size())
        throw std::out_of_range("Index out of range");
    mData[index] /= value;
}

typename Tensor::ShapeVector Tensor::shape()
{
    return mShape;
}

size_t Tensor::shape(const size_t &index) const
{
    if (index >= mShape.size())
        throw std::out_of_range("Tensor::shape: Index out of range");
    return mShape[index]; // return the shape at the given index
}

std::uint32_t Tensor::dimensionality() const
{
    return mShape.size(); // return the dimensionality
}

std::uint32_t Tensor::capacity()
{
    return mData.size(); // return the capacity
}

void Tensor::resize(const ShapeVector &dimensionality)
{
    mData.resize(std::accumulate(dimensionality.begin(), dimensionality.end(), 1, std::multiplies<>())); // resize the data vector
    mShape = dimensionality;                                                                                   // set the new shape
}

void Tensor::reshape(const ShapeVector &dimensionality)
{
    if (std::accumulate(dimensionality.begin(), dimensionality.end(), 1, std::multiplies<>()) != mData.size())
        throw std::invalid_argument("Tensor::reshape: New dimensionality does not match the capacity of the tensor");
    mShape = dimensionality; // set the new shape
}