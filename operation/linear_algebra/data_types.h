#ifndef DATA_TYPES_INCLUDE_GUARD
#define DATA_TYPES_INCLUDE_GUARD
// this is a template and not finished 
#include <vector>

class DATATYPE
{

};

template <typename T>
class SCALAR : public DATATYPE
{
    T __value;
public:
    SCALAR(T value) : __value(value){};
    T get_value(){return __value;};
};

template <typename T>
class VECTOR : public DATATYPE
{
    std::vector<T> __value;
public:
    VECTOR() : __value(){};
    VECTOR(std::vector<T> value) : __value(value){};
    std::vector<T> * get_value(){return &__value;};
};


template <typename T>
class MATRIX : public DATATYPE
{
    std::vector<std::vector<T>> __value;
public:
    MATRIX() : __value(){};
    MATRIX(std::vector<std::vector<T>> value) : __value(value){};
    std::vector<std::vector<T>> * get_value(){return &__value;};
};

#endif // DATA_TYPES_INCLUDE_GUARD