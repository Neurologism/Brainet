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

#endif // VECTOR_HPP
