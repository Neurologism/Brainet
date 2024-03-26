#ifndef PERCEPTRON_INCLUDE_GUARD
#define PERCEPTRON_INCLUDE_GUARD

#include<vector>
#include<random>
#include<stdexcept>
#include<functional>
#include<string>

class PERCEPTRON
{
private:
    std::vector<double> __weights;
    std::function<double(double)> __activation_function;
public:
    PERCEPTRON(int, std::string);
    ~PERCEPTRON();
    double calculate(std::vector<double>);
};

PERCEPTRON::PERCEPTRON(int weights, std::string activation_function)
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
{
    if(input.size() != __weights.size()-1)
    {
        throw std::invalid_argument("Dimensionality of Input and Weights do not match");
    }

    if(input[0] != 1)
    {
        throw std::invalid_argument("First element of input should be 1");
    }
    
    double value = __weights[0];
    for(int i=1;i < input.size();i++)
    {
        value += input[i-1] * __weights[i];
    }
    return value;
}

#endif