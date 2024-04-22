#ifndef DENSE_LAYER_INCLUDE_GUARD
#define DENSE_LAYER_INCLUDE_GUARD

#include "..\layer.h"
#include <vector>
#include <random>

/**
 * @brief Dense layer class, representing fully connected layers.
*/
class DENSE_LAYER : public LAYER
{
protected:
    int __input_size; // number of inputs to the layer
    int __output_size; // number of neurons in the layer
    std::vector<double> __weights; // weight matrix 
    std::vector<double> __bias; // bias vector
    std::vector<double> __bias_gradient;
    std::vector<double> __weights_gradient;
    std::vector<double> __activation_derivative;

    /**
     * @brief Activation function, which applies a non-linear transformation to the output of the layer.
     * @details This function should be implemented in the derived class. Applies a nonlinear transformation to each Element in input.
     * This function should be called after the linear transformation.
     * @note input is a reference and should be modified in place
    */

    virtual std::vector<double> activation_function(std::vector<double> input) =0;

    /**
     * @brief Derivative of the activation function.
     * @details This function should be implemented in the derived class. Applies the derivative of the activation function to each Element in input.
     * This function should be called after activation_function() and before backpropagation(). 
     * @note input is a reference and should be modified in place
    */
    virtual std::vector<double> differentiate_activation_function(std::vector<double> input) =0;

    /**
     * @brief Linear transformation, which evaluates the Formula input * weights + bias.
    */
    std::vector<double> linear_transformation(std::vector<double>& input, bool bias=true);
    ~DENSE_LAYER();
    
public:
    DENSE_LAYER(int input_size, int output_size);
    
    std::vector<double> operation(std::vector<double>&);
    void differentiate();
    std::vector<double> backpropagation(std::vector<double>&);
};

DENSE_LAYER::DENSE_LAYER(int input_size, int output_size)
{
    //////////////////////////////////////////
    std::default_random_engine generator; // this is unifnished and should be replaced by a more sophisticated initialization
    std::normal_distribution<double> distribution(-0.01,0.01); // initialize weights to small random values
    for(int i = 0;i < input_size * output_size;i++) 
    {
        __weights.push_back(distribution(generator));
    }
    for(int i = 0;i < output_size;i++) 
    {
        __bias.push_back(distribution(generator));
    }
    //////////////////////////////////////////
    __bias_gradient.resize(output_size);
    __weights_gradient.resize(input_size * output_size);
    __input_size = input_size;
    __output_size = output_size;
}

DENSE_LAYER::~DENSE_LAYER()
{
    __weights.clear();
    __bias.clear();
    __bias_gradient.clear();
    __weights_gradient.clear();
}

/**
 * @brief Linear transformation, which evaluates the Formula input * weights + bias.
 * @attention This function is a placeholder and should be replaced by a more efficient implementation.
 * @param bias If true, the bias is added to the output.
*/
std::vector<double> DENSE_LAYER::linear_transformation(std::vector<double>& input, bool bias)
{
    std::vector<double> output;
    for(int i=0; i < __output_size; i++) // export later in seperate matrix class and overload operator
    {
        output.push_back(0);
        for(int j=0; j < __input_size; j++)
        {
            output[i] += input[j] * __weights[j + i * __input_size];
        }
        if(bias)output[i] += __bias[i];
    }
    return output;
}

/**
 * @brief Forward pass over the layer.
*/
std::vector<double> DENSE_LAYER::operation(std::vector<double>& input)
{
    __input.swap(input);
    return activation_function(linear_transformation(__input));;
}

void DENSE_LAYER::differentiate()
{
    __activation_derivative = differentiate_activation_function(linear_transformation(__input)); // recomputing the input to the activation function
}


std::vector<double> DENSE_LAYER::backpropagation(std::vector<double>& gradient)
{
    for(int i = 0; i < __output_size; i++)
    {
        gradient[i] *= __activation_derivative[i];
        __bias_gradient[i] += gradient[i];
        for(int j = 0; j < __input_size; j++) // export later in seperate matrix class and overload operator
        {
            __weights_gradient[j + i * __input_size] += gradient[i] * __input[j]; // multiply to get matrix of gradients
        }
    }
    gradient = linear_transformation(gradient, false); // compute the gradient for the next layer
    return gradient;
}

#endif // DENSE_LAYER_INCLUDE_GUARD