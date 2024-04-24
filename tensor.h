#ifndef TENSOR_INCLUDE_GUARD
#define TENSOR_INCLUDE_GUARD

#include <vector>
#include <stdexcept>

/**
 * @brief Tensor class is a wrapper class for std::vector<double>. It is used to store data in a tensor format.
*/
class TENSOR 
{
private:
    std::vector<TENSOR *> __elements;
    double __data = 0;
public:
    TENSOR(){};
    TENSOR(double data) : __data(data){};
    TENSOR(std::vector<TENSOR *> elements) : __elements(elements){};
    ~TENSOR();
    void push_back(TENSOR * element);
    void pop_back();
    TENSOR * operator[](int index);
    int size();
    void operator=(TENSOR * tensor);
    void set_data(double data);
    double get_data();
    bool is_scalar();
    
};

TENSOR::~TENSOR()
{
    for (TENSOR * element : __elements)
    {
        free(element);
    }
}

/**
 * @brief push back an element to the tensor
*/
void TENSOR::push_back(TENSOR * element)
{
    if (__data)
    {
        throw std::invalid_argument("TENSOR::push_back: Cannot push to scalar.");
    }
    __elements.push_back(element);
}

/**
 * @brief pop last element from the tensor
*/
void TENSOR::pop_back()
{
    if (__data)
    {
        throw std::invalid_argument("TENSOR::pop_back: Cannot pop from scalar.");
    }
    if (__elements.size() == 0)
    {
        throw std::invalid_argument("TENSOR::pop_back: Cannot pop from empty tensor.");
    }
    __elements.pop_back();
}

/**
 * @brief get element at index
*/
TENSOR * TENSOR::operator[](int index)
{
    if (index < 0 || index >= __elements.size())
    {
        throw std::invalid_argument("TENSOR::operator[]: Index out of bounds.");
    }
    return __elements[index];
}

/**
 * @brief get the size of the tensor
*/
int TENSOR::size()
{
    if (__data)
    {
        throw std::invalid_argument("TENSOR::size: Cannot get size of scalar.");
    }
    return __elements.size();
}

/**
 * @brief assign the data of the tensor
*/
void TENSOR::operator=(TENSOR * tensor)
{
    __elements = tensor->__elements;
    if (__data)
    {
        throw std::invalid_argument("TENSOR::operator=: Cannot assign tensor to scalar.");
    }
}

/**
 * @brief get the data of the scalar
*/
double TENSOR::get_data()
{
    if (__elements.size())
    {
        throw std::invalid_argument("TENSOR::get_data: Cannot get data of tensor with elements.");
    }
    return __data;
}

/**
 * @brief set the data of the scalar
*/

void TENSOR::set_data(double data)
{
    __data = data;
    if (__elements.size())
    {
        throw std::invalid_argument("TENSOR::set_data: Cannot set data of tensor with elements.");
    }
}

/**
 * @brief check if the tensor is a scalar
*/
bool TENSOR::is_scalar()
{
    return __elements.size() == 0;
}

#endif // TENSOR_INCLUDE_GUARD