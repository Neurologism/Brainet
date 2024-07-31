#ifndef TENSOR_INCLUDE_GUARD
#define TENSOR_INCLUDE_GUARD

#include "dependencies.h"

/**
 * @brief The tensor class is a implementation of a tensor. It is used to store data in a multidimensional array. To do this it uses a vector to store the data and a vector to store the shape of the tensor.
*/
template <class T>
class TENSOR
{
    std::vector<T> __data; // the data of the tensor
    std::vector<std::uint32_t> __shape; // the shape of the tensor

    /**
     * @brief This function performs error checks that should be done at the beginning of each function.
     */
    void error_check()
    {
        if (__data.size() != std::accumulate(__shape.begin(),__shape.end(),1, std::multiplies<std::uint32_t>()))
            throw std::invalid_argument("TENSOR::data: Data size does not match the dimensionality of the tensor");
    }

public:
    /**
     * @brief Construct an empty new TENSOR object.
     */
    TENSOR() = default;
    /**
     * @brief Construct a new TENSOR object.
     * @param dimensionality The dimensionality of the tensor.
     * @param value The value used to initialize all elements of the tensor.
     * @param random If true the tensor is initialized with random values.
     */
    TENSOR(std::vector<std::uint32_t> dimensionality, double value = 0, bool random = false);
    /**
     * @brief Copy constructor not allowed.
     */
    TENSOR(const TENSOR<T> & tensor)
    {
        throw std::invalid_argument("TENSOR::TENSOR: Copy constructor not allowed");
    }
    /**
     * @brief Copy assignment not allowed.
     */
    TENSOR & operator=(const TENSOR<T> & tensor)
    {
        throw std::invalid_argument("TENSOR::operator=: Copy assignment not allowed");
    }
    TENSOR(TENSOR<T> && tensor) = default;
    TENSOR & operator=(TENSOR<T> && tensor) = default;
    ~TENSOR() = default;
    
    /**
     * @brief This function is used to access the data of the tensor. To do so it uses a vector of indices.
     * @param index The indices of the element.
     * @return T The element at the given indices.
     */
    T at(std::vector<std::uint32_t> index);
    /**
     * @brief This function is used to set the value of an element in the tensor. To do so it uses a vector of indices.
     * @param index The indices of the element.
     * @param value The value to be set.
     */
    void set(std::vector<std::uint32_t> index, T value);
    /**
     * @brief This function returns the shape of the tensor.
     * @return std::vector<std::uint32_t> The shape of the tensor.
     */
    std::vector<std::uint32_t> shape(){return __shape;};
    /**
     * @brief This function returns the shape of the tensor at a given index.
     * @param index The index of the shape.
     * @return std::uint32_t The shape at the given index.
     */
    std::uint32_t shape(std::uint32_t index){return __shape[index];};
    /**
     * @brief This function returns the dimensionality of the tensor.
     * @return std::uint32_t The dimensionality of the tensor.
     */
    std::uint32_t dimensionality(){return __shape.size();};
    /**
     * @brief This function returns the total capacity of the tensor.
     * @return The size of the tensor.
     */
    std::uint32_t size(){return __data.size();};
    /**
     * @brief This function resizes the tensor.
     * @param dimensionality The new dimensionality of the tensor.
     */
    void resize(std::vector<std::uint32_t> dimensionality){__shape = dimensionality; __data.resize(std::accumulate(dimensionality.begin(),dimensionality.end(),1, std::multiplies<double>()));};
    /**
     * @brief This function returns the data of the tensor.
     * @return std::vector<T> The data of the tensor.
     */
    std::vector<T> & data(){error_check();return __data;};
    /**
     * @brief This function returns a transposed version of the tensor if the tensor is 2D.
     * @return std::shared_ptr<TENSOR<T>> The transposed tensor.
     */
    std::shared_ptr<TENSOR<T>> transpose();

    /**
     * @brief This function reshapes the tensor.
     * @param dimensionality The new dimensionality of the tensor.
     */
    void reshape(std::vector<std::uint32_t> dimensionality);
};

template <class T>
T TENSOR<T>::at(std::vector<std::uint32_t> index)
{
    if(index.size() != __shape.size())
        throw std::invalid_argument("TENSOR::at: Index size does not match the dimensionality of the tensor");
    std::uint32_t _block_size = std::accumulate(__shape.begin(), __shape.end(), 1, std::multiplies<std::uint32_t>()); // product of all dimensions
    std::uint32_t _index = 0;
    // calculate the index of the element
    for (std::uint32_t i = 0; i < index.size(); i++)
    {
        _block_size /= __shape[i];
        _index += index[i] * _block_size;
    }
    if(_index >= __data.size())
        throw std::out_of_range("Index out of range");
    return __data[_index]; // return the element
}


template <class T>
TENSOR<T>::TENSOR(std::vector<std::uint32_t> dimensionality, double value, bool random)
{
    __data = std::vector<T>(std::accumulate(dimensionality.begin(),dimensionality.end(),1, std::multiplies<std::uint32_t>()), value); // initialize the data vector
    if(random) // if random is true, initialize the tensor with random values
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1, 1);
        for(std::uint32_t i = 0; i < __data.size(); i++)
        {
            __data[i] = dis(gen);
        }
    }
    __shape = dimensionality; // set the shape of the tensor
}

template <class T>
void TENSOR<T>::set(std::vector<std::uint32_t> index, T value)
{
    if(index.size() != __shape.size())
        throw std::invalid_argument("TENSOR::set: Index size does not match the dimensionality of the tensor");
    std::uint32_t _block_size = std::accumulate(__shape.begin(), __shape.end(), 1, std::multiplies<std::uint32_t>()); // product of all dimensions
    std::uint32_t _index = 0;
    // calculate the index of the element
    for (std::uint32_t i = 0; i < index.size(); i++)
    {
        _block_size /= __shape[i];
        _index += index[i] * _block_size;
    }
    if(_index >= __data.size())
        throw std::out_of_range("Index out of range");
    __data[_index] = value; // set the element
}

template <class T>
std::shared_ptr<TENSOR<T>> TENSOR<T>::transpose()
{
    if(__shape.size() != 2)
        throw std::invalid_argument("TENSOR::transpose: Tensor must be 2D");
    std::shared_ptr<TENSOR<T>> _tensor = std::make_shared<TENSOR<T>>(TENSOR<T>({__shape[1],__shape[0]})); // create a new tensor with the transposed shape
    std::vector<T> _data(__data.size());
    // create a vector with the transposed data
    for(std::uint32_t i = 0; i < __shape[0]; i++)
    {
        for(std::uint32_t j = 0; j < __shape[1]; j++)
        {
            _data[j*__shape[0] + i] = __data[i*__shape[1] + j];
        }
    }
    _tensor->data() = _data; // set the data of the new tensor
    return _tensor; // return the new tensor
}

template <class T>
void TENSOR<T>::reshape(std::vector<std::uint32_t> dimensionality)
{
    if(std::accumulate(dimensionality.begin(),dimensionality.end(),1, std::multiplies<std::uint32_t>()) != __data.size())
        throw std::invalid_argument("TENSOR::reshape: New dimensionality does not match the size of the tensor");
    __shape = dimensionality; // set the new shape
}

#endif // TENSOR_INCLUDE_GUARD