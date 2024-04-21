#ifndef RECTIFIED_LINEAR_UNIT_INCLUDE_GUARD
#define RECTIFIED_LINEAR_UNIT_INCLUDE_GUARD

#include "dense_layer.h"

/**
 * @brief Rectified linear unit class, representing the ReLU activation function f(x) = max(x, 0).
*/
class ReLU : public DENSE_LAYER
{
protected:
    double __left_gradient;
public:
    ReLU(int input_size, int output_size);
    ~ReLU();
    std::vector<double> activation_function(std::vector<double>& input);
    std::vector<double> differentiate_activation_function(std::vector<double>& input);
};

/**
 * @brief Constructor for the ReLU class.
 * @param input_size The dimensionality of the input.
 * @param output_size The number of neurons.
*/
ReLU::ReLU(int input_size, int output_size) : DENSE_LAYER(input_size, output_size)
{
}

std::vector<double> ReLU::activation_function(std::vector<double>& input)
{
    for(int i=0; i < input.size(); i++)
    {
        input[i] = input[i] > 0 ? input[i] : __left_gradient * input[i];
    }
    return input;
}

std::vector<double> ReLU::differentiate_activation_function(std::vector<double>& input)
{
    for(int i=0; i < input.size(); i++)
    {
        input[i] = input[i] > 0 ? 1 : __left_gradient;
    }
    return input;
}


/**
 * @brief Leaky ReLU class, representing the activation function f(x) = max(x, 0) + left_gradient * min(x, 0).
*/
class LeakyReLU : public ReLU
{
public:
    LeakyReLU(int input_size, int output_size, double left_gradient);
    ~LeakyReLU();
};

/**
 * @brief Constructor for the LeakyReLU class.
 * @param input_size The dimensionality of the input.
 * @param output_size The number of neurons.
 * @param left_gradient The gradient of the function for x < 0.
*/
LeakyReLU::LeakyReLU(int input_size, int output_size, double left_gradient) : ReLU(input_size, output_size)
{
    __left_gradient = left_gradient;
}

/**
 * @brief Absolute ReLU class, representing the activation function f(x) = |x|.
*/
class AbsoluteReLU : public ReLU
{
public:
    AbsoluteReLU(int input_size, int output_size);
    ~AbsoluteReLU();
    std::vector<double> activation_function(std::vector<double>& input);
    std::vector<double> differentiate_activation_function(std::vector<double>& input);
};

/**
 * @brief Constructor for the AbsoluteReLU class.
 * @param input_size The dimensionality of the input.
 * @param output_size The number of neurons.
*/
AbsoluteReLU::AbsoluteReLU(int input_size, int output_size) : ReLU(input_size, output_size)
{
    __left_gradient = -1;
}

// add parametric RELU
// add Maxout

#endif // RECTIFIED_LINEAR_UNIT_INCLUDE_GUARD