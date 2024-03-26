#ifndef PERCEPTRON_INCLUDE_GUARD
#define PERCEPTRON_INCLUDE_GUARD

#include<vector>
#include<random>
#include<stdexcept>
#include<functional>
#include<string>
#include"activation_functions.h"

class PERCEPTRON
{
private:
    std::vector<double> __weights;
public:
    PERCEPTRON(int);
    ~PERCEPTRON();
    double calculate(std::vector<double>);
    void adapt_weights(double, double);
    void train(std::vector<std::vector<double>>);
};

PERCEPTRON::PERCEPTRON(int weights)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,0.1); // initialize weights to small random values
    for(int i = 0;i <= weights;i++) 
    {
        __weights.push_back(distribution(generator)); // bias
    }
}

PERCEPTRON::~PERCEPTRON()
{
    
}

double PERCEPTRON::calculate(std::vector<double> input)
/**
 * @attention Determines the dot product of input and the weights vector. Performs the Heaviside step
 * function as activation function.
*/
{
    using namespace activation_functions;
    if(input.size() != __weights.size()-1)
    {
        throw std::invalid_argument("Dimensionality of Input and Weights do not match");
    }

    if(input[0] != 1)
    {
        throw std::invalid_argument("First element of input should be 1");
    }
    
    double potential = __weights[0];
    for(int i=1;i < input.size();i++)
    {
        potential += input[i-1] * __weights[i];
    }
    return heaviside_step(potential);
}
/**
 * @attention adds amount * learning_rate to every weight
 * @param amount The error of the last forward pass
 * @param learning_rate The current learning rate of the network
*/
void PERCEPTRON::adapt_weights(double amount, double learning_rate)
{
    for(double & weight : __weights)
    {
        weight += amount * learning_rate;
    }
}



class PERCEPTRON_LAYER
{
private:
    int size;
public:
    PERCEPTRON_LAYER(int);
    ~PERCEPTRON_LAYER();
};

PERCEPTRON_LAYER::PERCEPTRON_LAYER(int size)
{
}

PERCEPTRON_LAYER::~PERCEPTRON_LAYER()
{
}

#endif