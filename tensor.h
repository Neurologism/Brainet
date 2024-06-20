#ifndef TENSOR_INCLUDE_GUARD
#define TENSOR_INCLUDE_GUARD

#include "dependencies.h"

/**
 * @brief TENSOR class is a wrapper class for a vector of doubles. It is used to store the data of a variable.
 * It is used to access a 1D vector as a multidimensional tensor.
*/
template <class T>
class TENSOR
{
    std::vector<T> __data;
    std::vector<int> __shape;
public:
    TENSOR(std::vector<int> dimensionality, bool random = false)
    {
        __shape = dimensionality;
        __data = std::vector<T>(std::accumulate(dimensionality.begin(),dimensionality.end(),1, std::multiplies<double>()), 0);
        if(random)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-0.1, 0.1);
            for(int i = 0; i < __data.size(); i++)
            {
                __data[i] = dis(gen);
            }
        }
    }
    T at(std::vector<int> index);
    void set(std::vector<int> index, T value);
    std::vector<int> shape(){return __shape;};
    int dimensionality(){return __shape.size();};
};

/**
 * @brief returns the value at the index
 * @param index vector of indices
 * @return the value at the index
*/
template <class T>
T TENSOR<T>::at(std::vector<int> index)
{
    int _block_size = std::accumulate(__shape.begin(), __shape.end(), 1, std::multiplies<int>());
    int _index = 0;
    for (int i = 0; i < index.size(); i++)
    {
        _block_size /= __shape[i];
        _index += index[i] * _block_size;
    }
    if(_index >= __data.size())
        throw std::out_of_range("Index out of range");
    return __data[_index];
}

/**
 * @brief sets the value at the index
 * @param index vector of indices
 * @param value the value to be set
*/
template <class T>
void TENSOR<T>::set(std::vector<int> index, T value)
{
    int _block_size = std::accumulate(__shape.begin(), __shape.end(), 1, std::multiplies<int>());
    int _index = 0;
    for (int i = 0; i < index.size(); i++)
    {
        _block_size /= __shape[i];
        _index += index[i] * _block_size;
    }
    if(_index >= __data.size())
        throw std::out_of_range("Index out of range");
    __data[_index] = value;
}

#endif // TENSOR_INCLUDE_GUARD