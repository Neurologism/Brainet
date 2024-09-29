//
// Created by servant-of-scietia on 9/28/24.
//

#include "datatypes/vector.hpp"

Vector::Vector(const ShapeVector &dimensionality) : Tensor(dimensionality)
{
    if (dimensionality.size() != 1)
        throw std::invalid_argument("Vector::Vector: Dimensionality must be 1.");
}

Vector::Vector(const ShapeVector &dimensionality, const Precision &value) : Tensor(dimensionality, value)
{
    if (dimensionality.size() != 1)
        throw std::invalid_argument("Vector::Vector: Dimensionality must be 1.");
}

Vector::Vector(const std::vector<Precision> &data) : Tensor({data.size()})
{
    this->mData = data;
}

Precision Vector::at(const std::uint32_t &i)
{
    if (i >= this->mData.size())
        throw std::invalid_argument("Vector::at: Index out of range.");

    return this->mData[i];
}

void Vector::set(const std::uint32_t &i, const Precision &value)
{
    if (i >= this->mData.size())
        throw std::invalid_argument("Vector::set: Index out of range.");

    this->mData[i] = value;
}

Precision Vector::dot(const Vector &other) // make more efficient in the future
{
    if (this->mData.size() != other.mData.size())
        throw std::invalid_argument("Vector::dot: Vectors must have the same size.");

    Precision result = 0;
    for (std::uint32_t i = 0; i < this->mData.size(); i++)
    {
        result += this->mData[i] * other.mData[i];
    }
    return result;
}