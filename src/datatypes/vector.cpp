//
// Created by servant-of-scietia on 20.09.24.
//

#include "datatypes/vector.hpp"

template <typename T>
Vector<T>::Vector(const ShapeVector &dimensionality) : Tensor<T>(dimensionality)
{
    if (dimensionality.size() != 1)
        throw std::invalid_argument("Vector::Vector: Dimensionality must be 1.");
}

template <typename T>
Vector<T>::Vector(const ShapeVector &dimensionality, const T &value) : Tensor<T>(dimensionality, value)
{
    if (dimensionality.size() != 1)
        throw std::invalid_argument("Vector::Vector: Dimensionality must be 1.");
}

template <typename T>
Vector<T>::Vector(const std::vector<T> &data) : Tensor<T>({data.size()})
{
    this->mData = data;
}

template <typename T>
T Vector<T>::at(const std::uint32_t &i)
{
    if (i >= this->mData.size())
        throw std::invalid_argument("Vector::at: Index out of range.");

    return this->mData[i];
}

template <typename T>
void Vector<T>::set(const std::uint32_t &i, const T &value)
{
    if (i >= this->mData.size())
        throw std::invalid_argument("Vector::set: Index out of range.");

    this->mData[i] = value;
}

template <typename T>
T Vector<T>::dot(const Vector<T> &other) // make more efficient in the future
{
    if (this->mData.size() != other.mData.size())
        throw std::invalid_argument("Vector::dot: Vectors must have the same size.");

    T result = 0;
    for (std::uint32_t i = 0; i < this->mData.size(); i++)
    {
        result += this->mData[i] * other.mData[i];
    }
    return result;
}