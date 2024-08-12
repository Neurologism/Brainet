#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "dependencies.hpp"

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

    {
        if (mData.size() != std::accumulate(mShape.begin(),mShape.end(),1, std::multiplies<ShapeType>()))
            throw std::invalid_argument("Tensor::data: DataVector size does not match the dimensionality of the tensor");
    }

public:
    /**
     * @brief Construct an empty new Tensor object.
     */
    Tensor() = default;

    /**
     * @brief Construct a new Tensor object. The tensor is initialized with a given value.
     * @param dimensionality The dimensionality of the tensor.
     * @param value The value to initialize the tensor with.
     */
    Tensor(ShapeVector dimensionality, T value);


    /**
     * @brief Construct a new Tensor object. The tensor is initialized with random values.
     * @param dimensionality The dimensionality of the tensor.
     */
    Tensor(ShapeVector dimensionality);

    ~Tensor() = default;
    
    /**
     * @brief This function is used to access the data of the tensor. To do so it uses a vector of indices.
     * @param index The indices of the element.
     * @return The element at the given position.
     */
    T at(ShapeVector index);

    /**
     * @brief This function is used to access the data of the tensor with a single index.
     * @param index The index of the element.
     * @return The element at the given index.
     */
    T at(ShapeType index);

    /**
     * @brief This function is used to set the value of an element in the tensor. To do so it uses a vector of indices.
     * @param index The indices of the element.
     * @param value The value to be set.
     */
    void set(ShapeVector index, T value);

    /**
     * @brief This function returns the shape of the tensor.
     * @return The shape of the tensor.
     */
    ShapeVector shape(){return mShape;};

    /**
     * @brief This function returns the shape of the tensor at a given index.
     * @param index The index of the shape.
     * @return The shape at the given index.
     */
    ShapeType shape(ShapeType index){return mShape[index];};

    /**
     * @brief This function returns the dimensionality of the tensor.
     * @return ShapeType The dimensionality of the tensor.
     */
    ShapeType dimensionality(){return mShape.size();};

    /**
     * @brief This function returns the total capacity of the tensor.
     * @return The size of the tensor.
     */
    ShapeType size(){return mData.size();};

    /**
     * @brief This function resizes the tensor.
     * @param dimensionality The new dimensionality of the tensor.
     */
    void resize(ShapeVector dimensionality);

    /**
     * @brief This function reshapes the tensor.
     * @param dimensionality The new dimensionality of the tensor.
     */
    void reshape(ShapeVector dimensionality);
};

template <class T>
T Tensor<T>::at(ShapeVector index)
{
    if(index.size() != mShape.size())
        throw std::invalid_argument("Tensor::at: Index size does not match the dimensionality of the tensor");

    ShapeType _block_size = std::accumulate(mShape.begin(), mShape.end(), 1, std::multiplies<ShapeType>()); // product of all dimensions
    ShapeType _index = 0;
    // calculate the index of the element
    for (ShapeType i = 0; i < index.size(); i++)
    {
        _block_size /= mShape[i];
        _index += index[i] * _block_size;
    }
    if(_index >= mData.size())
        throw std::out_of_range("Index out of range");
    return mData[_index]; // return the element
}

template <class T>
T Tensor<T>::at(ShapeType index)
{
    if(index >= mData.size())
        throw std::out_of_range("Index out of range");
    return mData[index]; // return the element
}

template <class T>
Tensor<T>::Tensor(ShapeVector dimensionality)
{
    mData = {};
    mShape = dimensionality;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0, 1.0E-8);
    for(ShapeType i = 0; i < mData.size(); i++)
    {
        mData.push_back(dis(gen));
    }
}

template <class T>
Tensor<T>::Tensor(ShapeVector dimensionality, T value)
{
    mData = DataVector(std::accumulate(dimensionality.begin(),dimensionality.end(),1, std::multiplies<ShapeType>()), value); // initialize the data vector
    if(random) // if random is true, initialize the tensor with random values
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dis(0, 0.0001);
        for(ShapeType i = 0; i < mData.size(); i++)
        {
            mData[i] = dis(gen);
        }
    }
    mShape = dimensionality; // set the shape of the tensor
}

template <class T>
void Tensor<T>::set(ShapeVector index, T value)
{
    if(index.size() != mShape.size())
        throw std::invalid_argument("Tensor::set: Index size does not match the dimensionality of the tensor");
    ShapeType _block_size = std::accumulate(mShape.begin(), mShape.end(), 1, std::multiplies<ShapeType>()); // product of all dimensions
    ShapeType _index = 0;
    // calculate the index of the element
    for (ShapeType i = 0; i < index.size(); i++)
    {
        _block_size /= mShape[i];
        _index += index[i] * _block_size;
    }
    if(_index >= mData.size())
        throw std::out_of_range("Index out of range");
    mData[_index] = value; // set the element
}

template <class T>
void Tensor<T>::reshape(ShapeVector dimensionality)
{
    if(std::accumulate(dimensionality.begin(),dimensionality.end(),1, std::multiplies<ShapeType>()) != mData.size())
        throw std::invalid_argument("Tensor::reshape: New dimensionality does not match the size of the tensor");
    mShape = dimensionality; // set the new shape
}

#endif // TENSOR_HPP