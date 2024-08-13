#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "../dependencies.hpp"

/**
 * @brief The tensor class is a implementation of a tensor. It is used to store data in a multidimensional array. To do this it uses a vector to store the data and a vector to store the shape of the tensor.
*/
template <class T>
class Tensor
{
protected:

    typedef std::vector<T> DataVector;
    typedef std::uint32_t ShapeType;
    typedef std::vector<ShapeType> ShapeVector;

    DataVector mData; // the data of the tensor
    ShapeVector mShape; // the shape of the tensor

    std::uint32_t calculateIndex(const ShapeVector & index);

public:
    /**
     * @brief Construct an empty new Tensor object.
     */
    Tensor() = default;
    
    /**
     * @brief Construct a new Tensor object. The tensor is initialized with random values.
     * @param dimensionality The dimensionality of the tensor.
     */
    Tensor(const ShapeVector & dimensionality);

    /**
     * @brief Construct a new Tensor object. The tensor is initialized with a given value.
     * @param dimensionality The dimensionality of the tensor.
     * @param value The value to initialize the tensor with.
     */
    Tensor(const ShapeVector & dimensionality, const T & value);
    
    ~Tensor() = default;
    
    /**
     * @brief This function is used to access the data of the tensor. To do so it uses a vector of indices.
     * @param index The indices of the element.
     * @return The element at the given position.
     */
    T at(const ShapeVector & index);

    /**
     * @brief This function is used to access the data of the tensor with a single index.
     * @param index The index of the element.
     * @return The element at the given index.
     */
    T at(const ShapeType & index);

    /**
     * @brief This function is used to set the value of an element in the tensor. To do so it uses a vector of indices.
     * @param index The indices of the element.
     * @param value The value to be set.
     */
    void set(const ShapeVector & index, const T & value);

    /**
     * @brief This function is used to set the value of an element in the tensor with a single index.
     * @param index The index of the element.
     * @param value The value to be set.
     */
    void set(const ShapeType & index, const T & value);

    /**
     * @brief This function is used to add a value to the tensor. To do so it uses a vector of indices.
     * @param index The indices of the element.
     * @param value The value to be added.
     */
    void add(const ShapeVector & index, const T & value);

    /**
     * @brief This function is used to add a value to the tensor with a single index.
     * @param index The index of the element.
     * @param value The value to be added.
     */
    void add(const ShapeType & index, const T & value);

    /**
     * @brief This function is used to subtract a value from the tensor. To do so it uses a vector of indices.
     * @param index The indices of the element.
     * @param value The value to be subtracted.
     */
    void subtract(const ShapeVector & index, const T & value);

    /**
     * @brief This function is used to subtract a value from the tensor with a single index.
     * @param index The index of the element.
     * @param value The value to be subtracted.
     */
    void subtract(const ShapeType & index, const T & value);

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
    ShapeType shape(const ShapeType & index);

    /**
     * @brief This function returns the dimensionality of the tensor.
     * @return The dimensionality of the tensor.
     */
    std::uint32_t dimensionality();

    /**
     * @brief This function returns the total capacity of the tensor.
     * @return The size of the tensor.
     */
    std::uint32_t capacity();

    /**
     * @brief This function resizes the tensor.
     * @param dimensionality The new dimensionality of the tensor.
     */
    void resize(const ShapeVector & dimensionality);

    /**
     * @brief This function reshapes the tensor.
     * @param dimensionality The new dimensionality of the tensor.
     */
    void reshape(const ShapeVector & dimensionality);
};

template <class T>
std::uint32_t Tensor<T>::calculateIndex(const ShapeVector & index)
{
    if(index.size() != mShape.size())
        throw std::invalid_argument("Tensor::calculateIndex: Index size does not match the dimensionality of the tensor");

    ShapeType _block_size = std::accumulate(mShape.begin(), mShape.end(), 1, std::multiplies<ShapeType>()); // product of all dimensions
    ShapeType _index = 0;

    // calculate the index of the element
    for (std::uint32_t i = 0; i < index.size(); i++)
    {
        _block_size /= mShape[i];
        _index += index[i] * _block_size;
    }

    if(_index >= mData.size())
        throw std::out_of_range("Index out of range");
    return _index;
}

template <class T>
Tensor<T>::Tensor(const ShapeVector & dimensionality)
{
    mData = {};
    mShape = dimensionality;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dis(0, 1.0E-8);
    for(std::uint32_t i = 0; i < mData.size(); i++)
    {
        mData.push_back(dis(gen));
    }
}

template <class T>
Tensor<T>::Tensor(const ShapeVector & dimensionality, const T & value)
{
    mData = DataVector(std::accumulate(dimensionality.begin(),dimensionality.end(),1, std::multiplies<ShapeType>()), value); // initialize the data vector
    mShape = dimensionality; // set the shape of the tensor
}


template <class T>
T Tensor<T>::at(const ShapeVector & index)
{
    return mData[calculateIndex(index)];
}

template <class T>
T Tensor<T>::at(const ShapeType & index)
{
    if(index >= mData.size())
        throw std::out_of_range("Index out of range");
    return mData[index];
}


template <class T>
void Tensor<T>::set(const ShapeVector & index, const T & value)
{
    mData[calculateIndex(index)] = value;
}

template <class T>
void Tensor<T>::set(const ShapeType & index, const T & value)
{
    if(index >= mData.size())
        throw std::out_of_range("Index out of range");
    mData[index] = value;
}

template <class T>
void Tensor<T>::add(const ShapeVector & index, const T & value)
{
    mData[calculateIndex(index)] += value;
}

template <class T>
void Tensor<T>::add(const ShapeType & index, const T & value)
{
    if(index >= mData.size())
        throw std::out_of_range("Index out of range");
    mData[index] += value;
}

template <class T>
void Tensor<T>::subtract(const ShapeVector & index, const T & value)
{
    mData[calculateIndex(index)] -= value;
}

template <class T>
Tensor<T>::ShapeVector Tensor<T>::shape()
{
    return mShape;
}

template <class T>
Tensor<T>::ShapeType Tensor<T>::shape(const ShapeType & index)
{
    if(index >= mShape.size())
        throw std::out_of_range("Tensor::shape: Index out of range");
    return mShape[index]; // return the shape at the given index
}

template <class T>
std::uint32_t Tensor<T>::dimensionality()
{
    return mShape.size(); // return the dimensionality
}

template <class T>
std::uint32_t Tensor<T>::capacity()
{
    return mData.size(); // return the capacity
}

template <class T>
void Tensor<T>::resize(const ShapeVector & dimensionality)
{
    mData.resize(std::accumulate(dimensionality.begin(),dimensionality.end(),1, std::multiplies<ShapeType>())); // resize the data vector
    mShape = dimensionality; // set the new shape
}

template <class T>
void Tensor<T>::reshape(const ShapeVector & dimensionality)
{
    if(std::accumulate(dimensionality.begin(),dimensionality.end(),1, std::multiplies<ShapeType>()) != mData.size())
        throw std::invalid_argument("Tensor::reshape: New dimensionality does not match the capacity of the tensor");
    mShape = dimensionality; // set the new shape
}

#endif // TENSOR_HPP