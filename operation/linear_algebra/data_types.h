#ifndef DATA_TYPES_INCLUDE_GUARD
#define DATA_TYPES_INCLUDE_GUARD

#include <vector>


class DATATYPE{};

class SCALAR : public DATATYPE
{
    double __value;
public:
    SCALAR(double value) : __value(value){};
    double get_value(){return __value;};
};

class VECTOR : public DATATYPE
{
    std::vector<double> __value;
public:
    VECTOR() : __value(){};
    VECTOR(std::vector<double> value) : __value(value){};
    std::vector<double> * get_value(){return &__value;};
};



class MATRIX : public DATATYPE
{
    std::vector<std::vector<double>> __value;
public:
    MATRIX() : __value(){};
    MATRIX(std::vector<std::vector<double>> value) : __value(value){};
    std::vector<std::vector<double>> * get_value(){return &__value;};
};

#endif // DATA_TYPES_INCLUDE_GUARD