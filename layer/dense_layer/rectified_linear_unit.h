#ifndef RECTIFIED_LINEAR_UNIT_INCLUDE_GUARD
#define RECTIFIED_LINEAR_UNIT_INCLUDE_GUARD

#include "dense_layer.h"

/**
 * @brief Rectified linear unit class, representing the ReLU activation function f(x) = max(x, 0).
*/
class ReLU : public DENSE_LAYER
{
public:
    ReLU(int input_size, int output_size);
    ~ReLU();
    std::vector<double> activation_function(std::vector<double>& input);
    std::vector<double> differentiate_activation_function(std::vector<double>& input);
};

ReLU::ReLU(int input_size, int output_size) : DENSE_LAYER(input_size, output_size)
{
}

ReLU::~ReLU()
{
}

std::vector<double> ReLU::activation_function(std::vector<double>& input)
{
    for(int i=0; i < input.size(); i++)
    {
        input[i] = input[i] > 0 ? input[i] : 0;
    }
    return input;
}

std::vector<double> ReLU::differentiate_activation_function(std::vector<double>& input)
{
    for(int i=0; i < input.size(); i++)
    {
        input[i] = input[i] > 0 ? 1 : 0;
    }
    return input;
}


/**
 * @brief Leaky rectified linear unit class, representing the Leaky ReLU activation function f(x) = max(x, 0.01x).
*/
class leaky_ReLU : public DENSE_LAYER
{
public:
    leaky_ReLU(int input_size, int output_size);
    ~leaky_ReLU();
    std::vector<double> activation_function(std::vector<double>& input);
    std::vector<double> differentiate_activation_function(std::vector<double>& input);
};

leaky_ReLU::leaky_ReLU(int input_size, int output_size) : DENSE_LAYER(input_size, output_size)
{
}

leaky_ReLU::~leaky_ReLU()
{
}

std::vector<double> leaky_ReLU::activation_function(std::vector<double>& input)
{
    for(int i=0; i < input.size(); i++)
    {
        input[i] = input[i] > 0 ? input[i] : 0.01 * input[i];
    }
    return input;
}

std::vector<double> leaky_ReLU::differentiate_activation_function(std::vector<double>& input)
{
    for(int i=0; i < input.size(); i++)
    {
        input[i] = input[i] > 0 ? 1 : 0.01;
    }
    return input;
}




#endif