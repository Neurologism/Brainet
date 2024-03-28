#ifndef PERCEPTRON_INCLUDE_GUARD
#define PERCEPTRON_INCLUDE_GUARD

#include<vector>
#include<random>
#include<stdexcept>
#include<functional>
#include<string>
#include<iostream>
#include"activation_functions.h"

class PERCEPTRON
{
private:
    std::vector<double> __weights;
public:
    PERCEPTRON(int);
    ~PERCEPTRON();
    double predict(std::vector<double>);
    void adapt_weights(double, double, bool);
    void train(std::vector<std::vector<double>>,std::vector<double>,int,double);
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

/**
 * @attention Determines the dot product of input and the weights vector. Performs the Heaviside step
 * function as activation function.
*/
double PERCEPTRON::predict(std::vector<double> input)
{
    using namespace activation_functions;
    if(input.size() != __weights.size()-1)
    {
        throw std::invalid_argument("Dimensionality of Input and Weights do not match");
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
void PERCEPTRON::adapt_weights(double amount, double learning_rate, bool print_weights=false)
{
    for(double & weight : __weights)
    {
        weight += amount * learning_rate;
        if(print_weights)std::cout<<weight<<' ';
    }
    if(print_weights)std::cout<<'\n';
}


/**
 * @attention Training process for a single perceptron classificator 
 * @param inputs The data matrix 
 * @param target The answer wished to predict for every row 
 * @param epochs The number of training iterations
 * @param learning_rate The step size of gradient descent 
*/
void PERCEPTRON::train(std::vector<std::vector<double>> inputs,std::vector<double> target, int epochs, double learning_rate)
{
    if(inputs.size() != target.size())
    {
        throw std::invalid_argument("Dimensions of Input Matrix and Target vector do not match");
    }
    for(int i=0;i < epochs;i++)
    {
        double error=0;
        for(int j=0;j < inputs.size(); j++)
        {
            error += target[j]-predict(inputs[j]); // difference between prediction and target value is counted in the eroor
        }
        adapt_weights(error,learning_rate, i%1==0);
        std::cout<<error<<'\n';
    }
}



class PERCEPTRON_LAYER
{
private:
    std::vector<PERCEPTRON>__layer;
    int __input_size;
public:
    PERCEPTRON_LAYER(int,int);
    ~PERCEPTRON_LAYER();
    std::vector<double> forward_pass(std::vector<double> &);
};

PERCEPTRON_LAYER::PERCEPTRON_LAYER(int size,int input_size) : __input_size(input_size)
{
    __layer.resize(size,PERCEPTRON(input_size));
}

PERCEPTRON_LAYER::~PERCEPTRON_LAYER()
{
}

/**
 * @attention performs simple vector matrix multiplication with the input vector and the weight matrix 
 * @param input The input to propagate through the layer 
*/
std::vector<double> PERCEPTRON_LAYER::forward_pass(std::vector<double> & input)
{
    std::vector<double> output;
    for(PERCEPTRON & perceptron : __layer)
    {
        output.push_back(perceptron.predict(input));
    }
    return output;
}

#endif