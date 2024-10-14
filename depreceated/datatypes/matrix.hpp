#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "tensor.hpp"

/**
 * @brief The Matrix class is a wrapper around the Tensor class that provides an extended interface for matrix operations.
 */
class Matrix : public Tensor
{
public:
    typedef std::vector<Precision> DataVector;
    typedef std::vector<size_t> ShapeVector;


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
    Matrix(const ShapeVector &dimensionality, const Precision &value);

    /**
     * @brief Construct a new Matrix object.
     * @param data The data to initialize the matrix with.
     */
    explicit Matrix(const std::vector<std::vector<Precision>> &data);

    ~Matrix() = default;

    /**
     * @brief Access the element at the given position.
     * @param i The row index.
     * @param j The column index.
     * @return The element at the given position.
     */
    Precision at(const std::uint32_t &i, const std::uint32_t &j);

    /**
     * @brief Set the element at the given position.
     * @param row The row index.
     * @param col The column index.
     * @param value The value to set the element to.
     */
    void set(const std::uint32_t &row, const std::uint32_t &col, const Precision &value);

    /**
     * @brief Get the data of the matrix.
     * @return The data of the matrix.
     */
    DataVector &getData();

    /**
     * @brief Get the shape of the matrix.
     * @return The shape of the matrix.
     */
    ShapeVector &getShape();

    /**
     * @brief Resize the matrix to a new shape.
     * @param rows The number of rows.
     * @param cols The number of columns.
     */
    void resize(const std::uint32_t &rows, const std::uint32_t &cols);
};

#endif // MATRIX_HPP