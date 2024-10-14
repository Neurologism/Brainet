#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "dependencies.hpp"
#include "config.hpp"


/**
 * @brief The tensor class is an implementation of a tensor.
 * It is used to store data in a multidimensional array.
 * To do this, it uses a vector to store the data and a vector to store the shape of the tensor.
 */
class Tensor
{
protected:
    typedef std::vector<Precision> DataVector;
    typedef std::vector<size_t> ShapeVector;

    DataVector mData;   // the data of the tensor
    ShapeVector mShape; // the shape of the tensor

    std::uint32_t calculateIndex(const ShapeVector &index);

public:
    /**
     * @brief Construct an empty new Tensor object.
     */
    Tensor() = default;

    /**
     * @brief Construct a new Tensor object. The tensor is initialized with random values.
     * @param dimensionality The dimensionality of the tensor.
     */
    explicit Tensor(const ShapeVector &dimensionality);

    /**
     * @brief Construct a new Tensor object. The tensor is initialized with a given value.
     * @param dimensionality The dimensionality of the tensor.
     * @param value The value to initialize the tensor with.
     */
    Tensor(const ShapeVector &dimensionality, const Precision &value);

    /**
     * @brief Construct a new Tensor object. The tensor is initialized with the data of another tensor.
     * @param tensor The tensor to copy the data from.
     */
    Tensor(const Tensor &tensor);

    Tensor& operator=(const Tensor &tensor);

    ~Tensor() = default;

    /**
     * @brief This function is used to access the data of the tensor. To do so, it uses a vector of indices.
     * @param index The indices of the element.
     * @return The element at the given position.
     */
    Precision at(const ShapeVector &index);

    /**
     * @brief This function is used to access the data of the tensor with a single index.
     * @param index The index of the element.
     * @return The element at the given index.
     */
    Precision at(const size_t &index);

    /**
     * @brief This function is used to set the value of an element in the tensor. To do so, it uses a vector of indices.
     * @param index The indices of the element.
     * @param value The value to be set.
     */
    void set(const ShapeVector &index, const Precision &value);

    /**
     * @brief This function is used to set the value of an element in the tensor with a single index.
     * @param index The index of the element.
     * @param value The value to be set.
     */
    void set(const size_t &index, const Precision &value);

    /**
     * @brief This function is used to add a value to the tensor. To do so, it uses a vector of indices.
     * @param index The indices of the element.
     * @param value The value to be added.
     */
    void add(const ShapeVector &index, const Precision &value);

    /**
     * @brief This function is used to add a value to the tensor with a single index.
     * @param index The index of the element.
     * @param value The value to be added.
     */
    void add(const size_t &index, const Precision &value);

    /**
     * @brief This function is used to subtract a value from the tensor. To do so, it uses a vector of indices.
     * @param index The indices of the element.
     * @param value The value to be subtracted.
     */
    void subtract(const ShapeVector &index, const Precision &value);

    /**
     * @brief This function is used to subtract a value from the tensor with a single index.
     * @param index The index of the element.
     * @param value The value to be subtracted.
     */
    void subtract(const size_t &index, const Precision &value);

    /**
     * @brief This function is used to multiply a value with the tensor. To do so, it uses a vector of indices.
     * @param index The indices of the element.
     * @param value The value to be multiplied.
     */
    void multiply(const ShapeVector &index, const Precision &value);

    /**
     * @brief This function is used to multiply a value with the tensor with a single index.
     * @param index The index of the element.
     * @param value The value to be multiplied.
     */
    void multiply(const size_t &index, const Precision &value);

    /**
     * @brief This function is used to divide the tensor by a value. To do so, it uses a vector of indices.
     * @param index The indices of the element.
     * @param value The value to be divided by.
     */
    void divide(const ShapeVector &index, const Precision &value);

    /**
     * @brief This function is used to divide the tensor by a value with a single index.
     * @param index The index of the element.
     * @param value The value to be divided by.
     */
    void divide(const size_t &index, const Precision &value);

    /**
     * @brief This function returns the shape of the tensor.
     * @return The shape of the tensor.
     */
    ShapeVector shape();

    /**
     * @brief This function returns the shape of the tensor at a given index.
     * @param index The index of the shape.
     * @return The shape at the given index.
     */
    [[nodiscard]] size_t shape(const size_t &index) const;

    /**
     * @brief This function returns the dimensionality of the tensor.
     * @return The dimensionality of the tensor.
     */
    [[nodiscard]] std::uint32_t dimensionality() const;

    /**
     * @brief This function returns the total capacity of the tensor.
     * @return The size of the tensor.
     */
    std::uint32_t capacity();

    /**
     * @brief This function resizes the tensor.
     * @param dimensionality The new dimensionality of the tensor.
     */
    void resize(const ShapeVector &dimensionality);

    /**
     * @brief This function reshapes the tensor.
     * @param dimensionality The new dimensionality of the tensor.
     */
    void reshape(const ShapeVector &dimensionality);
};

#endif // TENSOR_HPP