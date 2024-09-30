//
// Created by servant-of-scietia on 9/28/24.
//
#include "datatypes/matrix.hpp"

Matrix::Matrix(const ShapeVector &dimensionality) : Tensor(dimensionality)
{
    if (dimensionality.size() != 2)
        throw std::invalid_argument("Matrix::Matrix: Matrix must have exactly two dimensions.");
}

Matrix::Matrix(const ShapeVector &dimensionality, const Precision &value) : Tensor(dimensionality, value)
{
    if (dimensionality.size() != 2)
        throw std::invalid_argument("Matrix::Matrix: Matrix must have exactly two dimensions.");
}

Matrix::Matrix(const std::vector<std::vector<Precision>> &data) : Tensor({data.size(), data[0].size()}, 0)
{
    for (std::uint32_t i = 0; i < data.size(); i++)
    {
        for (std::uint32_t j = 0; j < data[0].size(); j++)
        {
            this->mData[i * this->mShape[1] + j] = data[i][j];
        }
    }
}

Precision Matrix::at(const std::uint32_t &i, const std::uint32_t &j)
{
    return mData[i * mShape[1] + j];
}

void Matrix::set(const std::uint32_t &i, const std::uint32_t &j, const Precision &value)
{
    if (i >= this->mShape[0] || j >= this->mShape[1])
        throw std::out_of_range("Matrix::set: Index out of range.");

    this->mData[i * this->mShape[1] + j] = value;
}

Matrix::DataVector& Matrix::getData()
{
    return this->mData;
}

Matrix::ShapeVector &Matrix::getShape()
{
    return this->mShape;
}

void Matrix::resize(const std::uint32_t &rows, const std::uint32_t &cols)
{
    this->mShape = {rows, cols};
    this->mData.resize(rows * cols);
}