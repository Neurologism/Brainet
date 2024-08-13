#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "tensor.hpp"

/**
 * @brief The Vector class is a wrapper around the Tensor class that provides a more intuitive interface for vector operations.
 */
template <typename T>
class Vector : public Tensor<T>
{
    typedef std::vector<T> DataVector;
    typedef std::vector<size_t> ShapeVector;

public:
    Vector() = default;

    /**
     * @brief Construct a new Vector object. The vector is initialized with random values.
     * @param dimensionality The dimensionality of the vector.
     */
    Vector(const ShapeVector &dimensionality);

    /**
     * @brief Construct a new Vector object. The vector is initialized with a given value.
     * @param dimensionality The dimensionality of the vector.
     * @param value The value to initialize the vector with.
     */
    Vector(const ShapeVector &dimensionality, const T &value);

    /**
     * @brief Construct a new Vector object.
     * @param data The data to initialize the vector with.
     */
    Vector(const std::vector<T> &data);

    ~Vector() = default;

    /**
     * @brief Access the element at the given position.
     * @param i The index of the element.
     * @return The element at the given position.
     */
    T at(const std::uint32_t &i);

    /**
     * @brief Set the element at the given position.
     * @param i The index of the element.
     * @param value The value to set the element to.
     */
    void set(const std::uint32_t &i, const T &value);

    /**
     * @brief Calculate the dot product of two vectors.
     * @param other The other vector to calculate the dot product with.
     * @return The dot product of the two vectors.
     */
    T dot(const Vector<T> &other);
};

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

#endif // VECTOR_HPP
