#include"deep_feedforward_networks/perceptron.h"
#include<iostream>

int main()
{
    PERCEPTRON p(5,"ReLU");
    std::cout<<p.calculate({1,1,1,1,1})<<'\n';
    
}