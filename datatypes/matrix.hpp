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
    typedef std::uint32_t ShapeType;
    typedef std::vector<ShapeType> ShapeVector;

public:
    Matrix() = default;

    /**
     * @brief Construct a new Matrix object. The matrix is initialized with random values.
     * @param dimensionality The dimensionality of the matrix.
     */
    Matrix(const ShapeVector & dimensionality) : Tensor<T>(dimensionality);

    /**
     * @brief Construct a new Matrix object. The matrix is initialized with a given value.
     * @param dimensionality The dimensionality of the matrix.
     * @param value The value to initialize the matrix with.
     */
    Matrix(const ShapeVector & dimensionality, const T & value) : Tensor<T>(dimensionality, value);

    /**
     * @brief Construct a new Matrix object.
     * @param data The data to initialize the matrix with.
     */
    Matrix(const std::vector<std::vector<T>> & data) : Tensor<T>({data.size(), data[0].size()}, 0);

    ~Matrix() = default;

    /**
     * @brief Access the element at the given position.
     * @param i The row index.
     * @param j The column index.
     * @return The element at the given position.
     */
    T at(const std::uint32_t & i, const std::uint32_t & j);

    /**
     * @brief Set the element at the given position.
     * @param i The row index.
     * @param j The column index.
     * @param value The value to set the element to.
     */
    void set(const std::uint32_t & i, const std::uint32_t & j, const T & value);

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
    std::shared_ptr<Matrix<T>> dot(const Matrix<T> & other);
};

template <typename T>
Matrix<T>::Matrix(const ShapeVector & dimensionality) : Tensor<T>(dimensionality)
{
    if (dimensionality.size() != 2)
        throw std::invalid_argument("Matrix::Matrix: Matrix must have exactly two dimensions.");
}

template <typename T>
Matrix<T>::Matrix(const ShapeVector & dimensionality, const T & value) : Tensor<T>(dimensionality, value)
{
    if (dimensionality.size() != 2)
        throw std::invalid_argument("Matrix::Matrix: Matrix must have exactly two dimensions.");
}

template <typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>> & data) : Tensor<T>({data.size(), data[0].size()}, 0)
{
    for (std::uint32_t i = 0; i < data.size(); i++)
    {
        for (std::uint32_t j = 0; j < data[0].size(); j++)
        {
            mData[i * mShape[1] + j] = data[i][j];
        }
    }
}

template <typename T>
T Matrix<T>::at(const std::uint32_t & i, const std::uint32_t & j)
{
    if (i >= mShape[0] || j >= mShape[1])
        throw std::out_of_range("Matrix::at: Index out of range.");
    
    return mData[i * mShape[1] + j];
}

template <typename T>
void Matrix<T>::set(const std::uint32_t & i, const std::uint32_t & j, const T & value)
{
    if (i >= mShape[0] || j >= mShape[1])
        throw std::out_of_range("Matrix::set: Index out of range.");

    mData[i * mShape[1] + j] = value;
}

template <class T>
std::shared_ptr<Matrix<T>> Matrix<T>::transpose()
{
    if(mShape.size() != 2)
        throw std::invalid_argument("Matrix::transpose: Matrix must have exactly two dimensions.");

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

template <class T>
std::shared_ptr<Matrix<T>> Matrix<T>::dot(const Matrix<T> & other) // move from matmul to here
{
    throw std::invalid_argument("Matrix::dot: Not implemented.");
    return nullptr;
}

#endif // MATRIX_HPP