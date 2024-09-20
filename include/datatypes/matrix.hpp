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
    explicit Matrix(const ShapeVector &dimensionality);

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

#endif // MATRIX_HPP