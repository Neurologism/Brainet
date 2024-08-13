#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "tensor.hpp"

/**
 * @brief The Matrix class is a wrapper around the Tensor class that provides an extended interface for matrix operations.
 */
template <typename T>
class Matrix : public Tensor<T>
{
    typedef std::vector<T> DataVector;
    typedef std::vector<size_t> ShapeVector;

public:
    Matrix() = default;

    /**
     * @brief Construct a new Matrix object. The matrix is initialized with random values.
     * @param dimensionality The dimensionality of the matrix.
     */
    Matrix(const ShapeVector &dimensionality);

    /**
     * @brief Construct a new Matrix object. The matrix is initialized with a given value.
     * @param dimensionality The dimensionality of the matrix.
     * @param value The value to initialize the matrix with.
     */
    Matrix(const ShapeVector &dimensionality, const T &value);

    /**
     * @brief Construct a new Matrix object.
     * @param data The data to initialize the matrix with.
     */
    Matrix(const std::vector<std::vector<T>> &data);

    ~Matrix() = default;

    /**
     * @brief Access the element at the given position.
     * @param i The row index.
     * @param j The column index.
     * @return The element at the given position.
     */
    T at(const std::uint32_t &i, const std::uint32_t &j);

    /**
     * @brief Set the element at the given position.
     * @param i The row index.
     * @param j The column index.
     * @param value The value to set the element to.
     */
    void set(const std::uint32_t &i, const std::uint32_t &j, const T &value);

    /**
     * @brief Transpose the matrix.
     * @return A new matrix that is the transposed version of the original matrix.
     */
    std::shared_ptr<Matrix<T>> transpose();

    /**
     * @brief Calculate the dot product of two matrices.
     * @param other The other matrix to calculate the dot product with.
     * @return The dot product of the two matrices.
     */
    std::shared_ptr<Matrix<T>> dot(const Matrix<T> &other);
};

template <typename T>
Matrix<T>::Matrix(const ShapeVector &dimensionality) : Tensor<T>(dimensionality)
{
    if (dimensionality.size() != 2)
        throw std::invalid_argument("Matrix::Matrix: Matrix must have exactly two dimensions.");
}

template <typename T>
Matrix<T>::Matrix(const ShapeVector &dimensionality, const T &value) : Tensor<T>(dimensionality, value)
{
    if (dimensionality.size() != 2)
        throw std::invalid_argument("Matrix::Matrix: Matrix must have exactly two dimensions.");
}

template <typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>> &data) : Tensor<T>({data.size(), data[0].size()}, 0)
{
    for (std::uint32_t i = 0; i < data.size(); i++)
    {
        for (std::uint32_t j = 0; j < data[0].size(); j++)
        {
            this->mData[i * this->mShape[1] + j] = data[i][j];
        }
    }
}

template <typename T>
T Matrix<T>::at(const std::uint32_t &i, const std::uint32_t &j)
{
    if (i >= this->mShape[0] || j >= this->mShape[1])
        throw std::out_of_range("Matrix::at: Index out of range.");

    return this->mData[i * this->mShape[1] + j];
}

template <typename T>
void Matrix<T>::set(const std::uint32_t &i, const std::uint32_t &j, const T &value)
{
    if (i >= this->mShape[0] || j >= this->mShape[1])
        throw std::out_of_range("Matrix::set: Index out of range.");

    this->mData[i * this->mShape[1] + j] = value;
}

template <class T>
std::shared_ptr<Matrix<T>> Matrix<T>::transpose()
{
    if (this->mShape.size() != 2)
        throw std::invalid_argument("Matrix::transpose: Matrix must have exactly two dimensions.");

    std::shared_ptr<Matrix<T>> transposedMatrix = std::make_shared<Matrix<T>>(Matrix<T>({this->mShape[1], this->mShape[0]}, 0));

    for (std::uint32_t i = 0; i < this->mShape[0]; i++)
    {
        for (std::uint32_t j = 0; j < this->mShape[1]; j++)
        {
            transposedMatrix->set(j, i, this->at(i, j));
        }
    }

    return transposedMatrix;
}

template <class T>
std::shared_ptr<Matrix<T>> Matrix<T>::dot(const Matrix<T> &other) // move from matmul to here
{
    throw std::invalid_argument("Matrix::dot: Not implemented.");
    return nullptr;
}

#endif // MATRIX_HPP