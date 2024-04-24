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

public:
    TENSOR(){};
    TENSOR(std::vector<TENSOR *> elements) : __elements(elements){};
    ~TENSOR();
    void push_back(TENSOR * element);
    void pop_back();
    TENSOR * operator[](int index);
    int size();
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
    __elements.push_back(element);
}

/**
 * @brief pop last element from the tensor
*/
void TENSOR::pop_back()
{
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
    return __elements.size();
}



/**
 * @brief Scalar class is a wrapper class for double. It is used to store data in a scalar format.
*/
class SCALAR : private TENSOR
{
    double __data;
public:
    SCALAR(double data) : __data(data){};
    double get_data();
    void set_data(double data);
};

/**
 * @brief get the data of the scalar
*/
double SCALAR::get_data()
{
    return __data;
}

/**
 * @brief set the data of the scalar
*/

void SCALAR::set_data(double data)
{
    __data = data;
}

#endif // TENSOR_INCLUDE_GUARD