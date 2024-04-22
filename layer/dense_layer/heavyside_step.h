#ifndef HEAVYSIDE_STEP_INCLUDE_GUARD
#define HEAVYSIDE_STEP_INCLUDE_GUARD

#include "dense_layer.h"

/**
 * @brief Heavyside step function class, representing the activation function f(x) = 1 if x > 0, 0 otherwise.
*/

class HeavysideStep : public DENSE_LAYER
{
public:
    HeavysideStep(int input_size, int output_size);
    ~HeavysideStep();
    std::vector<double> activation_function(std::vector<double> input);
    std::vector<double> differentiate_activation_function(std::vector<double> input);
};

/**
 * @brief Constructor for the HeavysideStep class.
 * @param input_size The dimensionality of the input.
 * @param output_size The number of neurons.
*/
HeavysideStep::HeavysideStep(int input_size, int output_size) : DENSE_LAYER(input_size, output_size)
{
}

std::vector<double> HeavysideStep::activation_function(std::vector<double> input)
{
    for(int i=0; i < input.size(); i++)
    {
        input[i] = input[i] > 0 ? 1 : 0;
    }
    return input;
}

std::vector<double> HeavysideStep::differentiate_activation_function(std::vector<double> input)
{
    for(int i=0; i < input.size(); i++)
    {
        input[i] = 0;
    }
    return input;
}

#endif // HEAVYSIDE_STEP_INCLUDE_GUARD