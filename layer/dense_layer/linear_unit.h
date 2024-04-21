#ifndef LINEAR_UNIT_INCLUDE_GUARD
#define LINEAR_UNIT_INCLUDE_GUARD

#include "dense_layer.h"

class Linear_Unit : public DENSE_LAYER
{
public:
    Linear_Unit(int input_size, int output_size);
    ~Linear_Unit();
    std::vector<double> activation_function(std::vector<double>& input);
    std::vector<double> differentiate_activation_function(std::vector<double>& input);
};

Linear_Unit::Linear_Unit(int input_size, int output_size) : DENSE_LAYER(input_size, output_size)
{
}

std::vector<double> Linear_Unit::activation_function(std::vector<double>& input)
{
    return input;
}

std::vector<double> Linear_Unit::differentiate_activation_function(std::vector<double>& input)
{
    for(int i=0; i < input.size(); i++)
    {
        input[i] = 1;
    }
    return input;
}

#endif // LINEAR_UNIT_INCLUDE_GUARD