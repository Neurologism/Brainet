#ifndef LAYER_INCLUDE_GUARD
#define LAYER_INCLUDE_GUARD

#include <vector>

/**
 * @brief Base class determining the properties each layer should have.
*/
class LAYER
{
protected:
    std::vector<double> __input; 

public:
    
    LAYER(){};
    ~LAYER();
    virtual std::vector<double> operation(std::vector<double>&) =0; // forward pass over layer
    virtual void differentiate() =0;   // calculate part of the chain rule
    virtual std::vector<double> backpropagation(std::vector<double>&) =0; // backward pass over the layer 
};


/**
 * @brief Dense layer class, representing fully connected layers.
*/
class DENSE_LAYER : public LAYER
{
protected:
    std::vector<double> __weights; // weight matrix 
    std::vector<double> __bias; // bias vector
    std::vector<double> __output;
    std::vector<double> __bias_gradient;
    std::vector<double> __weights_gradient;

    /**
     * @brief Activation function, which applies a non-linear transformation to the output of the layer.
     * @details This function should be implemented in the derived class. Applies a nonlinear transformation to each Element in __output.
    */
    virtual void activation_function() =0;

    /**
     * @brief Derivative of the activation function.
     * @details This function should be implemented in the derived class. Applies the derivative of the activation function to each Element in __input 
     * and stores the results in __gradient.
    */
    virtual void activation_derivative() =0;

public:
    DENSE_LAYER(int input_size, int output_size);
    ~DENSE_LAYER();
    std::vector<double> operation(std::vector<double>&);
    void differentiate();
    std::vector<double> backpropagation(std::vector<double>&);
};

DENSE_LAYER::DENSE_LAYER(int input_size, int output_size)
{
    __weights.resize(input_size * output_size);
    __bias.resize(output_size);
    __output.resize(output_size);
    __bias_gradient.resize(output_size);
}

std::vector<double> DENSE_LAYER::operation(std::vector<double>& input)
{
    __input.swap(input);
    for(int i=0; i < __output.size(); i++) // export later in seperate matrix class and overload operator
    {
        __output[i] = 0;
        for(int j=0; j < __input.size(); j++)
        {
            __output[i] += __input[j] * __weights[j + i * __input.size()];
        }
        __output[i] += __bias[i];
    }
    activation_function();
    return __output;
}

void DENSE_LAYER::differentiate()
{
    activation_derivative();
}


std::vector<double> DENSE_LAYER::backpropagation(std::vector<double>& gradient)
{
    for(int i = 0; i < __input.size(); i++)
    {
        __input[i] *= gradient[i];
        __bias_gradient[i] += __input[i];
    }

}

#endif